using TemplateMatchingCLI
using Documenter

DocMeta.setdocmeta!(TemplateMatchingCLI, :DocTestSetup, :(using TemplateMatchingCLI); recursive=true)

makedocs(;
    modules=[TemplateMatchingCLI],
    authors="Stefano Campanella <15182642+stefanocampanella@users.noreply.github.com> and contributors",
    repo="https://github.com/stefanocampanella/TemplateMatchingCLI.jl/blob/{commit}{path}#{line}",
    sitename="TemplateMatchingCLI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://stefanocampanella.github.io/TemplateMatchingCLI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stefanocampanella/TemplateMatchingCLI.jl",
    devbranch="master",
)
