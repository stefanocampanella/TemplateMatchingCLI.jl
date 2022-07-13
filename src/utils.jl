function fpsize2fptype(precision)
    if precision == 16
        type = Float16
    elseif precision == 32
        type = Float32
    elseif precision == 64
        type = Float64
    else
        throw(ArgumentError("Precision not supported"))
    end
    type
end

function cuttemplate(data, sensorscoordinates, template, data_starttime, freq_MHz, speed, window, eltype)
    originsample = round(Int, freq_MHz * (template.datetime - data_starttime).value)
    templatecoordinates = Vector(template[[:north, :east, :up]])
    head_len, tail_len = window
    templatedata = Stream{eltype}() 
    templateoffsets = Offsets()
    for (channel, (series, sensorcoordinates)) in zipdicts(data, sensorscoordinates)
        displacement = sensorcoordinates .- templatecoordinates
        distance = Base.splat(hypot)(displacement)
        offset = round(Int, freq_MHz * (distance / speed))
        arrivalsample = originsample + offset
        templatedata[channel] = series[arrivalsample - head_len: arrivalsample + tail_len]
        templateoffsets[channel] = offset
    end
    templatedata, templateoffsets
end

zipdicts(dicts...) = Dict(key => tuple((d[key] for d in dicts)...) for key in intersectkeys(dicts...))

intersectkeys(dicts...) = Base.splat(collect ∘ intersect)(map(keys, dicts)) 

function detect!(peaks_chnl, data, templates, tolerance, threshold, rel_distance, npeaksmax; iscudafunctional=false)
    progressbar = Progress(length(templates); output=stderr, enabled=!is_logging(stderr), showspeed=true)
    for template in templates
        if !any(map(ismissing, r))
            signal = computesignal(data, template, tolerance; iscudafunctional)
            distance = rel_distance * maximum(length, values(template.data))
            peaks, heights = TemplateMatching.findpeaks(signal, threshold, distance)
            if !(isempty(peaks) || length(peaks) > npeaksmax)
                put!(peaks_chnl, (template, peaks, heights))
            end
        end
        next!(progressbar)
    end
end

is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

function computesignal(data::Stream{T}, template, tolerance; iscudafunctional=false):: OffsetVector{T, Vector{T}} where {T <: AbstractFloat} 
    channels = intersectkeys(data, template.data, template.offsets)
    data_vec = dict2array(data, channels)
    templatedata_vec = dict2array(template.data, channels)
    offsets_vec = dict2array(template.offsets, channels)
    data_vec_d = uploaddata(data_vec, iscudafunctional) 
    templatedata_vec_d = uploaddata(templatedata_vec, iscudafunctional) 
    signal = TemplateMatching.correlatetemplate(data_vec_d, templatedata_vec_d, offsets_vec, tolerance, T; usefft=true)
    signal_p = parent(signal)
    signal_p .= abs.(signal_p .- median(signal_p))
    signal_p ./= median(signal_p)
    signal 
end

dict2array(d, keys) = [d[key] for key in keys]

uploaddata(data, iscudafunctional=false) = iscudafunctional ? CuArray.(data) : data

function process!(detections_chnl, peaks_chnl, data, sensors, head_len, freq, speed, tolerance, ccmin, nchmin)
    for (template, peaks, heights) in peaks_chnl
        detections = DataFrame()
        detections.peak_sample = peaks .+ head_len
        detections.peak_height = heights
        detections.template .= template.index
        detectionsdata = Vector{TemplateMatchEventData}(undef, length(peaks))
        Threads.@threads for k in eachindex(detectionsdata)
            detectionsdata[k] = processdetection(data, 
                                                 template, 
                                                 sensors,
                                                 peaks[k],
                                                 freq,
                                                 head_len,
                                                 speed,
                                                 tolerance,
                                                 ccmin, 
                                                 nchmin)
        end
        if !isempty(detections)
            put!(detections_chnl, hcat(detections, DataFrame(detectionsdata)))
        end
    end
end

function processdetection(data, template, sensors, peak, freq, head_len, speed, tolerance, ccmin, nchmin)
    commonchannels = intersectkeys(data, template.data, template.offsets, sensors)
    subsample_estimates = Dict(key => TemplateMatching.estimatetoa(data[key], 
                                                                   template.data[key], 
                                                                   peak + template.offsets[key], 
                                                                   tolerance) 
                               for key in commonchannels)
    filter!(p -> p.second[2] > ccmin, subsample_estimates)
    channels = collect(keys(subsample_estimates))
    guess = [template.north, template.east, template.up, (peak + head_len) / freq]
    if length(channels) >= nchmin
        sensors_vec = [sensors[key] for key in channels]
        toas = [(sample + head_len) / freq for (sample, _) in values(subsample_estimates)]
        candidate = locate(vcat.(sensors_vec, toas), speed, guess)
        crosscorrelation = mean(cc for (_, cc) in values(subsample_estimates))
        relative_magnitude = magnitude(data, template, peak, channels)
        if Optim.converged(candidate)
            north, east, up, origin_time = candidate.minimizer
            multilateration_residual = candidate.minimum
        else
            north, east, up, origin_time = guess
            multilateration_residual = missing
        end
    else
        north, east, up, origin_time = guess
        multilateration_residual = missing
        crosscorrelation = missing
        relative_magnitude = missing
    end
    (; north, east, up, origin_time, magnitude=template.magnitude + relative_magnitude, crosscorrelation, channels, multilateration_residual)
end

locate(sensors_readings_itr, v, guess) = optimize(xt -> residue_rms(xt, sensors_readings_itr, v), guess)

residue_rms(xt, sensors_readings_itr, v) = sqrt(mean(ys -> line_element(ys - xt, v)^2, sensors_readings_itr))

function line_element(xt, v)
    x = view(xt, 1:3)
    t = xt[4]
    dot(x, x) - v^2 * t^2
end

function magnitude(data, template, peak, channels)
    data_vec = [data[key] for key in channels]
    template_vec = [template.data[key] for key in channels]
    offsets_vec = [template.offsets[key] for key in channels]
    TemplateMatching.magnitude(data_vec, template_vec, peak .+ offsets_vec)
end

function store(detections, outputpath)
    if isempty(detections)
        @info "No match found."
    else
        augmented_catalogue = reduce(vcat, detections)
        @info "Found $(nrow(augmented_catalogue)) matches."
        @info "Saving augmented catalogue at $outputpath." augmented_catalogue
        jldsave(outputpath; augmented_catalogue)
    end
end