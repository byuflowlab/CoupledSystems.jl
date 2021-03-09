using CoupledSystems
using Documenter

makedocs(;
    modules=[CoupledSystems],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide.md",
        "Theory" => "theory.md",
        "Library" => "library.md"
    ],
    sitename="CoupledSystems.jl",
    authors="Taylor McDonnell <taylor.golden.mcdonnell@gmail.com> and contributors",
)

deploydocs(
    repo = "github.com/byuflowlab/CoupledSystems.jl.git",
)
