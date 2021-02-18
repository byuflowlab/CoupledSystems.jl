using CoupledSystems
using Documenter

DocMeta.setdocmeta!(CoupledSystems, :DocTestSetup, :(using CoupledSystems); recursive=true)

makedocs(;
    modules=[CoupledSystems],
    authors="Taylor McDonnell <taylor.golden.mcdonnell@gmail.com> and contributors",
    repo="https://github.com/byuflowlab/CoupledSystems.jl/blob/{commit}{path}#{line}",
    sitename="CoupledSystems.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
