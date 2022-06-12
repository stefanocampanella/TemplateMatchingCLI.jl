is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


zipdicts(dicts...) = Dict(key => tuple((d[key] for d in dicts)...) for key in intersectkeys(dicts...))


intersectkeys(dicts...) = Base.splat(collect ∘ intersect)(map(keys, dicts)) 


collectbatch(xs, k, N) = N > 1 ? [x for (n, x) in enumerate(xs) if mod(n, N) == k] : collect(xs)


dict2array(d, keys) = [d[key] for key in keys]


residue_rms(xt, sensors_readings_itr, v) = sqrt(mean(ys -> line_element(ys - xt, v)^2, sensors_readings_itr))


locate(sensors_readings_itr, v, guess) = optimize(xt -> residue_rms(xt, sensors_readings_itr, v), guess)


function line_element(xt, v)
    x = view(xt, 1:3)
    t = xt[4]
    dot(x, x) - v^2 * t^2
end


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
    pre, post = window
    templatedata = Stream{eltype}() 
    templateoffsets = Offsets()
    for (channel, (series, sensorcoordinates)) in zipdicts(data, sensorscoordinates)
        displacement = sensorcoordinates .- templatecoordinates
        distance = Base.splat(hypot)(displacement)
        offset = round(Int, freq_MHz * (distance / speed))
        arrivalsample = originsample + offset
        templatedata[channel] = series[arrivalsample - pre: arrivalsample + post]
        templateoffsets[channel] = offset
    end
    templatedata, templateoffsets
end


function uploaddata(data::Stream{T}, gpus::Vector{CuDevice}, templatespergpu) where {T <: AbstractFloat}
    if isempty(gpus)
        data
    else
        data_d = similar(gpus, MultiDeviceStream{T})
        for (n, g) in enumerate(gpus)
            device!(g)
            data_d[n] = g, Semaphore(templatespergpu), Dict(key => CuArray(series) for (key, series) in data)
        end
        data_d
    end
end


@inline function correlate(data, template, offsets, tolerance, element_type; usefft=true)
    channels = intersectkeys(data, template, offsets)
    data_vec = dict2array(data, channels)
    template_vec = dict2array(template, channels)
    offsets_vec = dict2array(offsets, channels)
    TemplateMatching.correlatetemplate(data_vec, template_vec, offsets_vec, tolerance, element_type, usefft=usefft)
end


function computesignal(devicedata::Vector{MultiDeviceStream{T}}, template, tolerance) where {T <: AbstractFloat}
    gpu, semaphore, cudata = devicedata[Threads.threadid() % length(devicedata) + 1]
    device!(gpu)
    acquire(semaphore)
    signal = nothing
    while isnothing(signal)
        try
            cutemplate_data = Dict(key => CuArray(T.(series)) for (key, series) in template.data)
            cusignal = correlate(cudata, cutemplate_data, template.offsets, tolerance, T)
            let p = parent(cusignal)
                p .= abs.(p .- median(p))
                p ./= median(p)
            end
            signal = convert(OffsetVector{T, Vector{T}}, cusignal)
            CUDA.unsafe_free!(parent(cusignal))
            for series in values(cutemplate_data)
                CUDA.unsafe_free!(series)
            end
        catch err
            if err isa CuError
                @warn "An exception occurred while computing cross-correlation." gpu err
            else
                throw(err)
            end
        finally
            release(semaphore)
        end
    end
    signal
end


function computesignal(data::Dict{Int, Vector{T}}, template, tolerance) where {T <: AbstractFloat}
    signal = correlate(data, template.data, template.offsets, tolerance, T)
    signal .= abs.(signal .- median(signal))
    signal ./= median(signal)
    signal
end


function processdetections(data, template, sensors, peaks, heights, delay, freq, speed, tolerance, ccmin, nchmin)
    detections = DataFrame()
    detections.peak_sample = peaks .+ delay
    detections.peak_height = heights
    detections.template .= template.index
    detectionsdata = Vector{TemplateMatchEventData}(undef, length(peaks))
    Threads.@threads for k in eachindex(detectionsdata)
        detectionsdata[k] = processdetection(data, 
                                             template, 
                                             sensors,
                                             peaks[k],
                                             freq,
                                             delay,
                                             speed,
                                             tolerance,
                                             ccmin, 
                                             nchmin)
    end
    hcat(detections, DataFrame(detectionsdata))
end

function processdetection(data, template, sensors, peak, freq, delay, speed, tolerance, ccmin, nchmin)
    commonchannels = intersectkeys(data, template.data, template.offsets, sensors)
    subsample_estimates = Dict(key => TemplateMatching.estimatetoa(data[key], 
                                                                   template.data[key], 
                                                                   peak + template.offsets[key], 
                                                                   tolerance) 
                               for key in commonchannels)
    filter!(p -> p.second[2] > ccmin, subsample_estimates)
    channels = collect(keys(subsample_estimates))
    guess = [template.north, template.east, template.up, (peak + delay) / freq]
    if length(channels) >= nchmin
        sensors_vec = [sensors[key] for key in channels]
        toas = [(sample + delay) / freq for (sample, _) in values(subsample_estimates)]
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


function magnitude(data, template, peak, channels)
    data_vec = [data[key] for key in channels]
    template_vec = [template.data[key] for key in channels]
    offsets_vec = [template.offsets[key] for key in channels]
    TemplateMatching.magnitude(data_vec, template_vec, peak .+ offsets_vec)
end