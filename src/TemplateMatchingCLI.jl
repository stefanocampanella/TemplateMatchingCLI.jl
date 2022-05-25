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
using Comonicon

include("types.jl")
include("datetimeus.jl")
include("readers.jl")
include("utils.jl")
include("commands.jl")

@main

end
