"""
Read binary files, de-trend, filter, resample, and finally save data on disk. 
Binary files should be named <`datetime`>_<`experiment``>_ch<NN>&<MM>.bin and located 
all in the same directory `inputdirpath`.

# Arguments

- `inputdirpath`: directory of the binary files.
- `datetime`: datetime of data to be read.
- `experiment`: experiment identifier of data to be read.
- `outputdirpath`: path of the output directory.

# Options

- `-p, --precision`: FP precision to use for computation and storage.
- `-f, --freq`: original sampling frequency.
- `-l, --lopassfreq`: lower frequency in bandpass filter.
- `-h, --hipassfreq`: higher frequency in bandpass filter.
- `-n, --numpoles`: number of poles in Butterworth filter.
- `-r, --resamplefactor`: resampling factor.
- `-e, --exclude`: channels to exclude.
"""
@cast function preprocess(dirpath, datetime, experiment, outputpath; 
                          precision::Int=32, freq::Int=10_000_000, 
                          lopassfreq::Int=50_000, hipassfreq::Int=300_000, 
                          numpoles::Int=4, resamplefactor::Int=5, 
                          exclude::AbstractString="")
    bad_channels = isempty(exclude) ? [] : map(s -> parse(Int, s), split(exclude, ","))
    @info "Reading files from $dirpath (experiment: $experiment, time: $datetime)..."
    data = readlabdir(dirpath, datetime, experiment, bad_channels, typemax(Int), fptype(precision))
    if isempty(data)
        @warn "No data found"
    else
        responsetype = Bandpass(lopassfreq, hipassfreq, fs=freq)
        designmethod = Butterworth(numpoles)
        progressbar = Progress(length(data); output=stderr, enabled=!is_logging(stderr))
        @info "Pre-processing data..."
        Threads.@threads for n in collect(keys(data))
            if n ∉ bad_channels
                ys = data[n]
                xs = axes(data[n], 1)
                beta = (mean(xs .* ys) - mean(xs) * mean(ys)) / std(xs, corrected=false)
                alpha = mean(ys) - beta * mean(xs)
                ys_detrended = @. ys - alpha - beta * xs
                ys_filtered = filtfilt(digitalfilter(responsetype, designmethod), ys_detrended)
                ys_resampled = resample(ys_filtered, 1 //resamplefactor)
                data[n] = ys_resampled
                next!(progressbar)
            end
        end
        @info "Saving data..."
        starttime = DateTime(datetime, dateformat"yyyy-mm-dd_HH-MM-SS")
        resampledfreq = freq // resamplefactor
        endtime = starttime + Second(round(Int, minimum(length, values(data)) / resampledfreq))
        jldsave(joinpath(outputpath, "$(datetime)_$experiment.jld2"); data, starttime, endtime, freq=resampledfreq)
    end
end


"""
Cut templates.

# Arguments

- `datadirpath`: path of the directory of JLD2 data files.
- `sensorsxyzpath`: path of the CSV containing sensors coordinates.
- `cataloguepath`: path of the CSV catalogue of templates.
- `experiment`: name of the experiment of which load the data.
- `outputpath`: path of the output file.

# Options

- `-p, --precision`: FP precision to use for template storage.
- `-s, --speed`: P-wave speed in cm/us.
- `-w, --window`: template window in samples.
"""
@cast function maketemplates(datadirpath, experiment, sensorsxyzpath, cataloguepath, outputpath; 
                             precision=32, speed=0.67, window=(100, 500))
    @info "Reading catalogue..."
    catalogue = readcatalogue(cataloguepath)
    @info "Reading sensors coordinates..."
    sensorscoordinates = readsensorscoordinates(sensorsxyzpath)
    @info "Reading data and cutting templates..."
    re = Regex("\\Q$experiment\\E.jld2\$")
    eltype = fptype(precision)
    templates_data = Vector{MaybeTemplateData{eltype}}(missing, nrow(catalogue))
    templates_offsets = Vector{MaybeTemplateOffsets}(missing, nrow(catalogue))
    datapaths = collect(readdir(datadirpath, join=true))
    progressbar = Progress(length(datapaths); output=stderr, enabled=!is_logging(stderr))
    Threads.@threads for datapath in datapaths
        if !isnothing(match(re, datapath))
            dataset = load(datapath)
            data = dataset["data"]
            starttime_us = DateTimeMicrosecond(dataset["starttime"])
            endtime_us = DateTimeMicrosecond(dataset["endtime"])
            freq_MHz = round(Int, 1e-6 * dataset["freq"])
            templates_within_data = filter(r -> starttime_us <= r.datetime < endtime_us, catalogue)
            for template in eachrow(templates_within_data)
                template_data, offsets = cuttemplate(data,
                                                     sensorscoordinates,
                                                     template,
                                                     starttime_us, freq_MHz, speed, window, eltype)
                templates_data[template.index] = template_data
                templates_offsets[template.index] = offsets
            end
        end
        next!(progressbar)
    end
    @info "Saving templates"
    catalogue.data = templates_data
    catalogue.offsets = templates_offsets
    jldsave(joinpath(outputpath, "$experiment.jld2"); catalogue, speed, window)
end



"""
match templates.

# Arguments

- `datapath`: path of continuous data.
- `templatespath`: path of the directory of JLD2 data files.
- `sensorsxyzpath`: path of the CSV containing sensors coordinates.
- `outputpath`: path of the output file.

# Options

- `-p, --precision`: FP precision to use for computing crosscorrelations.
- `-t, --tolerance`: sample tolerance in stacking.
- `-h, --heightthreshold`: height threshold.
- `-d, --distance`: minimum distance between peaks.
- `-c, --correlationthreshold`: correlation threshold.
- `-n, --nchmin`: minimum number of channels.
- `-b, --batches`: batch to process.
"""
@cast function matchtemplates(datapath, templatespath, sensorspath, outputpath; 
                              precision=32, heightthreshold=0.4, distance=2, 
                              correlationthreshold=0.5, tolerance=5, nchmin=4,
                              batches="1/1")
    if CUDA.functional()
        gpus = CUDA.devices()
        num_gpus = length(gpu_list)
        @info "CUDA detected and functional, devices" gpu_list 
    else
        @info "CUDA not functional, using CPU"
    end
    @info "Reading data..."
    data, freq = load(datapath, "data", "freq")
    @info "Reading sensors coordinates..."
    sensors = readsensorscoordinates(sensorspath)
    @info "Reading templates..."
    catalogue, speed, window = load(templatespath, "catalogue", "speed", "window")
    filter!(r -> !any(map(ismissing, r)), catalogue)
    batch_number, total_batches = map(s -> parse(Int, s), split(batches, '/'))
    templates = collectbatch(Tables.namedtupleiterator(catalogue), batch_number, total_batches)
    delay, _  = window
    @info "Computing crosscorrelations and processing matches..."
    progressbar = Progress(length(templates); output=stderr, enabled=!is_logging(stderr))
    matches_vec = Vector{Union{DataFrame, Missing}}(undef, length(templates))
    Threads.@threads for n in eachindex(templates)
        if CUDA.functional()
            device!(gpus[n % num_gpus])
        end
        template = templates[n]
        crosscorrelation = correlate(data, template, tolerance, fptype(precision), direct=false)
        peaks, heights = TemplateMatching.findpeaks(crosscorrelation, heightthreshold, distance * (window[2] - window[1]))
        if isempty(peaks)
            matches_vec[n] = missing
        else
            matches = DataFrame()
            matches.peak_sample = peaks .+ delay
            matches.peak_height = heights
            matches.template .= template.index
            matches_data = Vector{TemplateMatchEventData}(undef, length(peaks))
            Threads.@threads for k in eachindex(matches_data)
                matches_data[k] = process_match(data, 
                                                template, 
                                                sensors,
                                                [template.north, template.east, template.up, (peaks[k] + delay) / freq],
                                                freq,
                                                delay,
                                                speed,
                                                tolerance,
                                                correlationthreshold, 
                                                nchmin)
            end
            matches_vec[n] = hcat(matches, DataFrame(matches_data))
        end
        next!(progressbar)
    end
    actual_matches =  skipmissing(matches_vec)
    if isempty(actual_matches)
        @info "No match found..."
    else
        @info "Saving augmented catalogue..."
        augmented_catalogue = reduce(vcat, actual_matches)
        filename = join(map(basename ∘ first ∘ splitext, [datapath, templatespath]), "_") * (total_batches > 1 ? "_$batch_number" : "") * ".jld2"
        jldsave(joinpath(outputpath, filename); augmented_catalogue)
    end
end