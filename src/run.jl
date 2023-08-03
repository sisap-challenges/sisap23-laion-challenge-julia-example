using SimilaritySearch, JLD2, CSV, Glob, LinearAlgebra
using Downloads: download

include("eval.jl")

"""
    download_data(url, outfile; verbose=false)

Download an url and save as the given file name
"""
function download_data(url, file; verbose=false)

    if isfile(file)
        println(stderr, "using local $file")
    else
        println(stderr, "downloading $url -> $file")
        mkpath(dirname(file))
        download(url, file; verbose)
    end

    file
end

"""
    build_searchgraph(dist::SemiMetric, db::AbstractDatabase, indexpath::String; verbose=false, minrecall=0.9)

Creates a `SearchGraph` index on the database `db`

- online documentation: <https://sadit.github.io/SimilaritySearch.jl/dev/>
- joss paper: _SimilaritySearch. jl: Autotuned nearest neighbor indexes for Julia
ES Tellez, G Ruiz - Journal of Open Source Software, 2022_ <https://joss.theoj.org/papers/10.21105/joss.04442.pdf>
- arxiv paper: 
```
Similarity search on neighbor's graphs with automatic Pareto optimal performance and minimum expected quality setups based on hyperparameter optimization
ES Tellez, G Ruiz - arXiv preprint arXiv:2201.07917, 2022
```
"""
function build_searchgraph(dist::SemiMetric, db::AbstractDatabase, indexpath::String; verbose=false, minrecall=0.9)
    algo = "SearchGraph"
    opt = MinRecall(minrecall)
    callbacks = SearchGraphCallbacks(opt)
    logbase = 1.5
    neighborhood = Neighborhood(; logbase)

    params = "r=$minrecall b=$logbase"
    indexname = joinpath(indexpath, "$algo-$params.jld2")
    isfile(indexname) && return indexname
    buildtime = @elapsed G = index!(SearchGraph(; db, dist, verbose); callbacks, neighborhood)
    optimtime = @elapsed optimize!(G, opt)
    meta = Dict(
        "buildtime" => buildtime,
        "matrix_size" => size(db.matrix),
        "optimtime" => optimtime,
        "algo" => algo,
        "params" => params
    )

    G, meta
end

"""
    run_search(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::AbstractString)

Solve `queries` with the give index (it will iterate on some parameter to find similar setups)

- `k` the number of nearest neighbors to retrieve
- `meta` metadata to be stored with results
- `resfile_` base name to create result files
"""
function run_search(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::AbstractString)
    resfile_ = replace(resfile_, ".h5" => "")
    step = 1.05f0
    delta = idx.search_algo.Δ / step^2
    params = meta["params"]

    while delta < 2f0
        idx.search_algo.Δ = delta
        resfile = "$resfile_-delta=$delta.h5"
        meta["params"] = "$params Δ=$(round(delta; digits=2))"
        run_search_(idx, queries, k, meta, resfile)
        delta *= step
    end
end

"""
    run_search(idx, queries::AbstractDatabase, k::Integer, meta, resfile::AbstractString)

Solve `queries` with the give index

- `k` the number of nearest neighbors to retrieve
- `meta` metadata to be stored with results
- `resfile` performance output file

"""
function run_search(idx, queries::AbstractDatabase, k::Integer, meta, resfile::AbstractString)
    run_search_(idx, queries, k, meta, resfile)
end

function run_search_(idx, queries::AbstractDatabase, k::Integer, meta, resfile::AbstractString)
    querytime = @elapsed knns, dists = searchbatch(idx, queries, k)
    jldsave(resfile;
        knns, dists,
        algo=meta["algo"],
        buildtime=meta["buildtime"] + meta["optimtime"],
        querytime,
        params=meta["params"],
        size=meta["size"],
        data=meta["data"]
    )
end

MIRROR = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
#MIRROR2 = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge/"

"""
    dbread(file, kind, key)


Loads a dataset stored in `file` and return it with its associated distance function, based on `kind` and `key` parameters.
"""
function dbread(file, kind, key)
    if startswith(kind, "clip768")
        @info "loading clip768 (converting Float16 -> Float32)"
        X = jldopen(file) do f
            Float32.(f["emb"])
        end

        for col in eachcol(X)
            normalize!(col)
        end

        return StrideMatrixDatabase(X), NormalizedCosineDistance()
    end
    
    X = jldopen(file) do f
        StrideMatrixDatabase(f[key])
    end

    if startswith(kind, "hamming")
        X, BinaryHammingDistance()
    elseif startswith(kind, "pca")
        X, SqL2Distance()
    else
        error("Unknown data $kind")
    end
end


"""
    main(kind, key, dbsize, k; outdir)

Runs an entire beenchmark

- `kind`: the kind of data (clip768, hamming, pca32, pca96)
- `key`: the key to access the database in each file (most use kind but clip768 will use "emb")
- `dbsize`: string denoting the size of the dataset ("100K", "300K", "10M", "30M", "100M"), million scale should not be used in GitHub Actions.
- `k`: the number of neighbors to find (official evaluation uses k=10, but you can use bigger values if your algorithm can take advantage of this)
"""
function main(kind, key, dbsize, k; outdir)
    queriesurl = "$MIRROR/public-queries-10k-$kind.h5"
    dataseturl = "$MIRROR/laion2B-en-$kind-n=$dbsize.h5"

    dfile = download_data(dataseturl, "data/$kind/$dbsize/dataset.h5")
    qfile = download_data(queriesurl, "data/$kind/query.h5")

    @info "loading $qfile and $dfile"
    db, dist = dbread(dfile, kind, key)
    queries, _ = dbread(qfile, kind, key)
    
    # loading or computing knns
    path = joinpath(outdir, kind)
    mkpath(path)
    @info "indexing, this can take a while!"
    G, meta = build_searchgraph(dist, db, path; verbose=true)
    mem = 0
    for l in G.adj
         mem += sizeof(l)
    end

    meta["size"] = dbsize
    meta["data"] = kind
    resfile = joinpath(path, "searchgraph-kind=$kind-size=$dbsize-k=$k")
    @info "searching"
    run_search(G, queries, k, meta, resfile)

    #=@info "running a bruteforce algorithm"
    meta["algo"] = "bruteforce"
    meta["buildtime"] = meta["optimtime"] = 0.0
    meta["params"] = "none"
    resfile = joinpath(path, "result-k=$k-bruteforce.h5")
    
    run_search(ExhaustiveSearch(; dist, db), queries, k, meta, resfile)
    =#
end

if !isinteractive()
    kind = get(ENV, "kind", "clip768v2")
    key = get(ENV, "key", "emb")
    # dbsize = get(ENV, "size", "300K")
    k = parse(Int, get(ENV, "k", "30"))
    
    for dbsize in ARGS
        outdir = joinpath("result", kind, dbsize)
        main(kind, key, dbsize, k; outdir)

        ### Please use the evaluation of https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation
        #=
        goldurl = "$MIRROR/laion2B-en-public-gold-standard-v2-$dbsize.h5"
        gfile = download_data(goldurl)
        
        res = evalresults(glob(joinpath(outdir, "*", "*.h5")), gfile, 10)
        CSV.write("results-$k-$dbsize.csv", res)
        =#
    end
end
