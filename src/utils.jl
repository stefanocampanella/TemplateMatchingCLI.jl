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


zipdicts(dicts...) = Dict(key => tuple((d[key] for d in dicts)...) for key in intersect((keys(d) for d in dicts)...))


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


function process_match(data_stream, template_stream, offsets_vec, guess, v_p, sensors_positions, correlation_threshold, nch_min)
    toas = estimatetoa.(data_stream, template_stream, offsets_vec, tolerance)
    readings = copy(sensors_positions)
    readings.index = 1:nrow(readings)
    readings.toas = [(sample + t_pre) / samplefreq for (sample, _) in toas]
    readings.cc = [cc for (_, cc) in toas]
    filter!(row -> row.cc > correlation_threshold, readings)
    crosscorrelation = mean(readings.cc)
    relative_magnitude = magnitude(data_stream[readings.index], template_stream[readings.index], offsets_vec[readings.index])
    nch = nrow(readings)
    if nch >= nch_min
        readings_matrix = eachrow(Matrix(readings[!, [:north, :east, :up, :toas]]))
        candidate = locate(readings_matrix, v_p, guess)
        north, east, up, origin_time = candidate.minimizer
        multilateration_residual = candidate.minimum
    else
        north, east, up, origin_time = guess
        multilateration_residual = missing
    end
    (; north, east, up, origin_time, relative_magnitude, crosscorrelation, nch, multilateration_residual)
end


function process_match(data_str, template_str, offsets_vec, guess, samplefreq, t_pre, v_p, sensors_positions, indx2ch, correlation_threshold, nch_min)
    estimates = estimatetoa.(data_str, template_str, offsets_vec, tolerance)
    toas = [(sample + t_pre) / samplefreq for (sample, _) in estimates]
    ccs = [cc for (_, cc) in estimates]
    indices = findall(>(correlation_threshold), ccs)
    crosscorrelation = mean(readings[indices])
    relative_magnitude = magnitude(data_str[indices], template_str[indices], offsets_vec[indices])
    nch = len(indices)
    if nch >= nch_min
        readings = [[sensors_positions[indx2ch[n]]; toas[n]] for n in indices]
        candidate = locate(readings, v_p, guess)
        north, east, up, origin_time = candidate.minimizer
        multilateration_residual = candidate.minimum
    else
        north, east, up, origin_time = guess
        multilateration_residual = missing
    end
    (; north, east, up, origin_time, relative_magnitude, crosscorrelation, nch, multilateration_residual)
end