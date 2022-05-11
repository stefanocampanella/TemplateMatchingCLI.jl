
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

function cuttemplate(data_stream, sensorscoordinates, template, freq, v_p, window)
    displacement = DataFrame()
    for s in [:north, :east, :up]
        displacement[!, s] = sensorscoordinates[!, s] .- template[s]
    end
    distances = map(Base.splat(hypot), eachrow(displacement[!, [:north, :east, :up]]))
    offsets = @. round(Int, freq * (distances / v_p))
    shifts = template[:sample] .+ offsets
    pre, post = window
    template_data = [series[shift - pre: shift + post] for (series, shift) in zip(data_stream, shifts)]
    template_data
end


function process_match(data_stream, template, match_offsets, guess, correlation_threshold, nch_min)
    toas = estimatetoa.(data_stream, template, match_offsets, tolerance)
    readings = copy(sensors_positions)
    readings.index = 1:nrow(readings)
    readings.toas = [(sample + t_pre) / samplefreq for (sample, _) in toas]
    readings.cc = [cc for (_, cc) in toas]
    filter!(row -> row.cc > correlation_threshold, readings)
    crosscorrelation = mean(readings.cc)
    relative_magnitude = magnitude(data_stream[readings.index], template[readings.index], match_offsets[readings.index])
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