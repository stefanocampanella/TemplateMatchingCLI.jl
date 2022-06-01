function fptype(precision)
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


is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


zipdicts(dicts...) = Dict(key => tuple((d[key] for d in dicts)...) for key in intersectkeys(dicts...))


intersectkeys(dicts...) = Base.splat(collect ∘ intersect)(map(keys, dicts)) 


collectbatch(xs, k, N) = N > 1 ? [x for (n, x) in enumerate(xs) if mod(n, N) == k] : collect(xs)


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


function correlate(data, template, tolerance, element_type; direct=false)
    tocuarray = CUDA.functional()
    channels = intersectkeys(data, template.data, template.offsets)
    data_vec = dict2array(data, channels, tocuarray)
    template_vec = dict2array(template.data, channels, tocuarray)
    offsets_vec = dict2array(template.offsets, channels)
    TemplateMatching.correlatetemplate(data_vec, template_vec, offsets_vec, tolerance, element_type, direct=direct)
end


dict2array(d, keys, tocuarray=false) = [tocuarray ? CuArray(d[key]) : d[key] for key in keys]


function process_match(data, template, sensors, guess, freq, delay, speed, tolerance, cc_threshold, nch_threshold)
    commonchannels = intersectkeys(data, template.data, template.offsets, sensors)
    subsample_estimates = Dict(key => TemplateMatching.estimatetoa(data[key], 
                                                                   template.data[key], 
                                                                   round(Int, guess[4] .+ template.offsets[key]), 
                                                                   tolerance) 
                               for key in commonchannels)
    filter!(p -> p.second[2] > cc_threshold, subsample_estimates)
    validchannels = collect(keys(subsample_estimates))
    if length(validchannels) >= nch_threshold
        sensors_vec = [sensors[key] for key in validchannels]
        toas = [(sample + delay) / freq for (sample, _) in values(subsample_estimates)]
        candidate = TemplateMatching.locate(vcat.(sensors_vec, toas), speed, guess)
        north, east, up, origin_time = candidate.minimizer
        multilateration_residual = candidate.minimum
        crosscorrelation = mean(cc for (_, cc) in values(subsample_estimates))
        data_vec = [data[key] for key in validchannels]
        template_vec = [template.data[key] for key in validchannels]
        offsets_vec = [template.offsets[key] for key in validchannels]
        relative_magnitude = TemplateMatching.magnitude(data_vec, template_vec, offsets_vec)
    else
        north, east, up, origin_time = guess
        multilateration_residual = missing
        crosscorrelation = missing
        relative_magnitude = missing
    end
    (; north, east, up, origin_time, magnitude=template.magnitude + relative_magnitude, crosscorrelation, validchannels, multilateration_residual)
end