Stream{T} = Dict{Int, Vector{T}} where T <: AbstractFloat

Offsets = Dict{Int, Int}

MaybeTemplateData{T} = Union{Missing, Stream{T}} where T <: AbstractFloat

MaybeTemplateOffsets = Union{Missing, Dict{Int, Int}}