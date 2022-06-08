module TemplateMatchingCLI

import TemplateMatching
using Dates
using CSV
using DataFrames
using StatsBase
using ProgressMeter
using DSP
using JLD2
using DataFrames
using Tables
using CUDA
using Comonicon
using Base: Semaphore, acquire, release
using Optim
using OffsetArrays

if CUDA.functional()
    @info "CUDA detected and functional." CUDA.version() CUDA.devices()
    const CUDA_DEVICES = collect(CUDA.devices())
else
    @info "CUDA not functional, using CPU."
    const CUDA_DEVICES = nothing
end

include("types.jl")
include("datetimeus.jl")
include("readers.jl")
include("utils.jl")
include("commands.jl")

@main

end
