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
using LinearAlgebra
using Tables
using CUDA
using Comonicon
using Optim
using OffsetArrays


include("types.jl")
include("datetimeus.jl")
include("io.jl")
include("detect.jl")
include("process.jl")
include("utils.jl")
include("commands.jl")

@main

end
