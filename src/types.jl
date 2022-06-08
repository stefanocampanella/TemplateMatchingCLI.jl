Stream{T} = Dict{Int, Vector{T}} where T <: AbstractFloat

DeviceStream{T} = Dict{Int, CuArray{T, 1, CUDA.Mem.DeviceBuffer}} where T <: AbstractFloat

MultiDeviceStream{T} = Vector{Tuple{CuDevice, Semaphore, DeviceStream{T}}} where T <: AbstractFloat

Offsets = Dict{Int, Int}

MaybeTemplateData{T} = Union{Missing, Stream{T}} where T <: AbstractFloat

MaybeTemplateOffsets = Union{Missing, Dict{Int, Int}}

TemplateMatchEventData = @NamedTuple begin
    north::Float64
    east::Float64
    up::Float64
    origin_time::Float64
    magnitude::Union{Missing, Float64}
    crosscorrelation::Union{Missing, Float64}
    validchannels::Vector{Int}
    multilateration_residual::Union{Missing, Float64}
end