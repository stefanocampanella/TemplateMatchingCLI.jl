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

include("types.jl")
include("datetimeus.jl")
include("readers.jl")
include("utils.jl")
include("commands.jl")

@main

end
