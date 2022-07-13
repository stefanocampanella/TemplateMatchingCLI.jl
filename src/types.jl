Stream{T} = Dict{Int, Vector{T}} where T <: AbstractFloat

MaybeTemplateData{T} = Union{Missing, Stream{T}} where T <: AbstractFloat

Offsets = Dict{Int, Int}

MaybeTemplateOffsets = Union{Missing, Dict{Int, Int}}

TemplateMatchEventData = @NamedTuple begin
    north::Float64
    east::Float64
    up::Float64
    origin_time::Dates.UTInstant{Microsecond}
    magnitude::Union{Missing, Float64}
    crosscorrelation::Union{Missing, Float64}
    channels::Vector{Int}
    multilateration_residual::Union{Missing, Float64}
end