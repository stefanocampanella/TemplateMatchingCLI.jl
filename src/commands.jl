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
    if precision == 16
        eltype = Float16
    elseif precision == 32
        eltype = Float32
    elseif precision == 64
        eltype = Float64
    else
        throw(ArgumentError("Precision not supported"))
    end
    @info "Reading files from $dirpath (experiment: $experiment, time: $datetime)..."
    data = readlabdir(dirpath, datetime, experiment, bad_channels, typemax(Int), eltype)
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
    newfreq = freq // resamplefactor
    endtime = starttime + Second(round(Int, minimum(length, values(data)) / newfreq))
    jldsave(joinpath(outputpath, "$(datetime)_$experiment.jld2"); data, starttime, endtime, freq=newfreq)
end


"""
Cut templates.

# Arguments

- `datapath`: path of JLD2 data file.
- `sensorsxyzpath`: path of the CSV containing sensors coordinates.
- `cataloguepath`: 
- `datetime`: datetime of data to be read.
- `outputdirpath`: path of the directory where to output template files.

# Options

- `-f, --freq`: original sampling frequency.
- `-w, --window`: template window in samples.
"""
@cast function maketemplates(datapath, sensorsxyzpath, cataloguepath, outputdirpath; 
                             v_p=0.67, window=(100, 500))
    @info "Loading data..."
    dataset = load(datapath)
    data = sort(dataset["data"])
    channels = keys(data)
    data_stream = values(data) 
    @info "Reading sensors coordinates..."
    sensorscoordinates = readsensorscoordinates(sensorsxyzpath, channels)
    @info "Reading catalogue..."
    starttime = dataset["starttime"]
    endtime = dataset["endtime"]
    freq = dataset["freq"]
    catalogue = readcatalogue(cataloguepath, round(Int, 1e-6 * freq), starttime, endtime)
    @info "Cutting and saving templates..."
    progressbar = Progress(length(data); output=stderr, enabled=!is_logging(stderr))
    for template in eachrow(catalogue)
        template_data = cuttemplate(data_stream, sensorscoordinates, template, round(Int, 1e-6 * freq), v_p, window)
        jldsave(joinpath(outputdirpath, "$(template.index).jld2"); template_data, template)
        next!(progressbar)
    end
end