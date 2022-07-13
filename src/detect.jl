function detect!(peaks_chnl, data, templates, tolerance, threshold, rel_distance, npeaksmax; iscudafunctional=false)
    progressbar = Progress(length(templates); output=stderr, enabled=!is_logging(stderr), showspeed=true)
    for template in templates
        if !any(map(ismissing, template))
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
