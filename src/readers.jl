function readlabfile(filepath, nb, eltype::Type{T}) where T <: AbstractFloat
    rawdata = read(filepath, nb)
    data = (convert(Vector{eltype}, ntoh.(reinterpret(Int16, rawdata))))
    data[1:2:end], data[2:2:end]
end


function readlabdir(dirpath, datetime, experiment, exclude_list, nb, eltype::Type{T}) where T <: AbstractFloat
    re = Regex("\\Q$datetime\\E_\\Q$experiment\\E_ch(?P<fst_channel>[0-9]+)&(?P<snd_channel>[0-9]+)\\.bin")
    data = Dict{Int, Vector{eltype}}()
    for filepath in readdir(dirpath, join=true)
        m = match(re, basename(filepath))
        if m !== nothing
            fst_channel, snd_channel = map(s -> parse(Int, m[s]), [:fst_channel, :snd_channel])
            if fst_channel ∉ exclude_list || snd_channel ∉ exclude_list
                fst_channel_data, snd_channel_data = readlabfile(filepath, nb, eltype)
                if fst_channel ∉ exclude_list
                    data[fst_channel] = fst_channel_data
                end
                if snd_channel ∉ exclude_list
                    data[snd_channel] = snd_channel_data
                end
            end
        end
    end
    data
end


function readcatalogue(filepath, samplefreq, starttime, endtime; 
                       columns = [:Year, :Month, :Day, :Hour, :Minute, :Second, :North, :East, :Up, :magnitude])
    df = CSV.read(filepath, DataFrame, select=columns)
    sec = floor.(Int, df.Second)
    usec = round.(Int, 1e6 * (df.Second - sec))
    catalogue = DataFrame()
    starttime_us = DateTimeMicrosecond(starttime)
    endtime_us = DateTimeMicrosecond(endtime)
    df.datetime = DateTimeMicrosecond.(df.Year, df.Month, df.Day, df.Hour, df.Minute, sec, usec)
    df.index = axes(df, 1)
    df = df[(@. starttime_us <= df.datetime < endtime_us), :]
    catalogue.sample = map(x -> round(Int, x.value * samplefreq), df.datetime .- starttime_us)
    catalogue.north = df.North
    catalogue.east = df.East
    catalogue.up = df.Up
    catalogue.index = df.index
    catalogue
end


function readsensorscoordinates(filepath, channels; header=[:north, :east, :up])
    coordinates = CSV.read(filepath, DataFrame; header)
    coordinates.sensor = axes(coordinates, 1) .- 1
    filter!(r -> r.sensor in channels, coordinates)
    coordinates
end

