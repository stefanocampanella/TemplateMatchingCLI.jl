function process!(detections_chnl, peaks_chnl, data, sensors, starttime, head_len, freq, speed, tolerance, ccmin, nchmin)
    for (template, peaks, heights) in peaks_chnl
        detections = DataFrame()
        detections.peak_sample = peaks .+ head_len
        detections.peak_height = heights
        detections.template .= template.index
        detectionsdata = Vector{TemplateMatchEventData}(undef, length(peaks))
        Threads.@threads for k in eachindex(detectionsdata)
            detectionsdata[k] = processdetection(
                data, template, sensors, starttime, peaks[k],
                freq, head_len, speed, tolerance, ccmin, nchmin)
        end
        if !isempty(detections)
            put!(detections_chnl, hcat(detections, DataFrame(detectionsdata)))
        end
    end
end

function processdetection(data, template, sensors, starttime, peak, freq, head_len, speed, tolerance, ccmin, nchmin)
    commonchannels = intersectkeys(data, template.data, template.offsets, sensors)
    subsample_estimates = Dict(
        key => TemplateMatching.estimatetoa(
            data[key], template.data[key], peak + template.offsets[key], tolerance) 
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
            north, east, up, timedelay = candidate.minimizer
            multilateration_residual = candidate.minimum
        else
            north, east, up, timedelay = guess
            multilateration_residual = missing
        end
    else
        north, east, up, timedelay = guess
        multilateration_residual = missing
        crosscorrelation = missing
        relative_magnitude = missing
    end
    (; north, east, up, origin_time=starttime + Microsecond(rount(Int, timedelay)), 
    magnitude=template.magnitude + relative_magnitude, crosscorrelation, channels, multilateration_residual)
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