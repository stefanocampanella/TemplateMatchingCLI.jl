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
- `-f, --freq`: original sampling frequency in Hz.
- `-l, --lopassfreq`: lower frequency in bandpass filter in Hz.
- `-h, --hipassfreq`: higher frequency in bandpass filter in Hz.
- `-n, --numpoles`: number of poles in Butterworth filter.
- `-r, --resamplefactor`: resampling factor.
- `-e, --exclude`: channels to exclude.
"""
@cast function preprocess(dirpath::AbstractString, datetime::AbstractString, 
                          experiment::AbstractString, outputpath::AbstractString; 
                          precision::Int=32, freq::Int=10_000_000, 
                          lopassfreq::Int=50_000, hipassfreq::Int=300_000, 
                          numpoles::Int=4, resamplefactor::Int=10, 
                          exclude::AbstractString="")
    bad_channels = isempty(exclude) ? [] : map(s -> parse(Int, s), split(exclude, ","))
    @info "Reading files from $dirpath (experiment: $experiment, time: $datetime)..."
    data = readlabdir(dirpath, datetime, experiment, bad_channels, typemax(Int), fpsize2fptype(precision))
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
        jldsave(outputpath; data, starttime, endtime, freq=resampledfreq)
    end
end

"""
Cut templates.

# Arguments

- `datadirpath`: path of the directory of JLD2 data files.
- `sensorsxyzpath`: path of the CSV containing sensors coordinates in cm.
- `cataloguepath`: path of the CSV catalogue of templates.
- `outputpath`: path of the output file.

# Options

- `-p, --precision`: FP size in bits,  precision to use for template storage.
- `-s, --speed`: P-wave speed in cm/us.
- `-w, --window`: template window in samples.
"""
@cast function maketemplates(datadirpath::AbstractString, sensorsxyzpath::AbstractString, 
                             cataloguepath::AbstractString, outputpath::AbstractString; 
                             precision::Int=32, speed::Float64=0.67, window::Tuple{Int, Int}=(50, 250))
    @info "Reading catalogue..."
    catalogue = readcatalogue(cataloguepath)
    @info "Reading sensors coordinates..."
    sensorscoordinates = readsensorscoordinates(sensorsxyzpath)
    @info "Reading data and cutting templates..."
    eltype = fpsize2fptype(precision)
    templates_data = Vector{MaybeTemplateData{eltype}}(missing, nrow(catalogue))
    templates_offsets = Vector{MaybeTemplateOffsets}(missing, nrow(catalogue))
    datapaths = collect(readdir(datadirpath, join=true))
    progressbar = Progress(length(datapaths); output=stderr, enabled=!is_logging(stderr))
    Threads.@threads for datapath in datapaths
        data, starttime, endtime, freq = load(datapath, "data", "starttime", "endtime", "freq")
        starttime_us = DateTimeMicrosecond(starttime)
        endtime_us = DateTimeMicrosecond(endtime)
        freq_MHz = 1e-6freq
        templates_within_data = filter(r -> starttime_us <= r.datetime < endtime_us, catalogue)
        for template in eachrow(templates_within_data)
            template_data, offsets = cuttemplate(
                data, sensorscoordinates, template,
                starttime_us, freq_MHz, speed, window, eltype)
            templates_data[template.index] = template_data
            templates_offsets[template.index] = offsets
        end
        next!(progressbar)
    end
    @info "Saving templates"
    catalogue.data = templates_data
    catalogue.offsets = templates_offsets
    jldsave(outputpath; catalogue, speed, window)
end

"""
Match templates.

# Arguments

- `datapath`: path of continuous data.
- `templatespath`: path of the directory of JLD2 data files.
- `sensorsxyzpath`: path of the CSV containing sensors coordinates in cm.
- `outputpath`: path of the output file.

# Options

- `--tolerance`: sample tolerance in stacking.
- `--threshold`: height threshold.
- `--distance`: minimum distance between peaks.
- `--ccmin`: correlation threshold.
- `--nchmin`: minimum number of channels.
- `--npeaksmax`: maximum number of detections to consider valid a template
"""
@cast function matchtemplates(datapath::AbstractString, templatespath::AbstractString, 
                              sensorspath::AbstractString, outputpath::AbstractString; 
                              tolerance::Int=8, threshold::Int=12, distance::Int=2, 
                              ccmin::Float64=0.5, nchmin::Int=4, npeaksmax::Int=1024)
    @info "Reading data from $(realpath(datapath))"
    data, starttime, freq = load(datapath, "data", "starttime", "freq")
    starttime_us = DateTimeMicrosecond(starttime)
    freq_MHz = 1e-6freq
    @info "Reading sensors coordinates from $(realpath(sensorspath))"
    sensors = readsensorscoordinates(sensorspath)
    @info "Reading templates from $(realpath(templatespath))"
    catalogue, speed, window = load(templatespath, "catalogue", "speed", "window")
    head_len, _ = window
    @info "Computing cross-correlations and processing matches"
    if CUDA.functional()
        @info "GPU acceleration available" CUDA.version()
        iscudafunctional = true
    else
        @info "GPU acceleration not available"
        iscudafunctional = false
    end
    templates = Tables.namedtupleiterator(catalogue)
    peaks_chnl = Channel{Tuple{eltype(templates), Vector{Int}, valtype(data)}}(
        chnl -> detect!(
            chnl, 
            data, templates, 
            tolerance, threshold, distance, npeaksmax; 
            iscudafunctional))
    detections_chnl = Channel{DataFrame}(
        chnl -> process!(
            chnl, 
            peaks_chnl, data, sensors, starttime_us,
            head_len, freq_MHz, speed, tolerance, ccmin, nchmin))
    savecatalogue(collect(detections_chnl), outputpath)
end