using SimilaritySearch, JLD2, CSV, Glob
using Downloads: download

include("eval.jl")

function download_data(url; verbose=false)
    file = joinpath("data", basename(url))

    if isfile(file)
        println(stderr, "using local $file")
    else
        println(stderr, "downloading $url -> $file")
        download(url, file; verbose)
    end

    file
end

function build_searchgraph(dist::SemiMetric, db::AbstractDatabase, indexpath::String; verbose=false, minrecall=0.9)
    algo = "SearchGraph"
    params = "MinRecall=$minrecall"
    indexname = joinpath(indexpath, "$algo-$params.jld2")
    isfile(indexname) && return indexname
    opt = MinRecall(minrecall)
    callbacks = SearchGraphCallbacks(opt)
    buildtime = @elapsed G = index!(SearchGraph(; db, dist, verbose); callbacks)
    optimtime = @elapsed optimize!(G, opt)
    meta = Dict(
        "buildtime" => buildtime,
        "matrix_size" => size(db.matrix),
        "optimtime" => optimtime,
        "algo" => algo,
        "params" => params
    )
    @info "saving the index"
    saveindex(indexname, G; meta)
    indexname
end

function run_search(idx, queries::AbstractDatabase, k::Integer, meta, resfile::AbstractString)
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

MIRROR = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"

function main(kind, dbsize, k, dist=SqL2Distance(); outdir)
    queriesurl = "$MIRROR/$kind/en-queries/public-queries-10k-$kind.h5"
    dataseturl = "$MIRROR/$kind/en-bundles/laion2B-en-$kind-n=$dbsize.h5"

    qfile = download_data(queriesurl)
    dfile = download_data(dataseturl)

    @info "loading $qfile and $dfile"
    queries = StrideMatrixDatabase(jldopen(f->f[kind], qfile))
    db = StrideMatrixDatabase(jldopen(f->f[kind], dfile))

    # loading or computing knns
    path = joinpath(outdir, kind)
    mkpath(path)
    @info "indexing, this can take a while!"
    indexname = build_searchgraph(dist, db, path)
    # Stop multithreading
    # here we will stop the multithreading vm and start a single core one
    @info "loading the index"
    G, meta = loadindex(indexname, db, staticgraph=true)
    meta["size"] = dbsize
    meta["data"] = kind
    resfile = joinpath(path, "result-k=$k-" * replace(basename(indexname), ".jld2" => "") * ".h5")
    @info "searching"
    run_search(G, queries, k, meta, resfile)

    @info "running a bruteforce algorithm"

    meta["algo"] = "bruteforce"
    meta["buildtime"] = meta["optimtime"] = 0.0
    meta["params"] = "none"
    resfile = joinpath(path, "result-k=$k-bruteforce.h5")
    run_search(ExhaustiveSearch(; dist, db), queries, k, meta, resfile)
end

for dbsize in ("100K", "300K")
    k = 30
    outdir = "out-$dbsize"
    #=
    main("hamming", dbsize, k, BinaryHammingDistance(); outdir)
    main("pca32", dbsize, k; outdir)
    main("pca96", dbsize, k; outdir)
=#
    prefix = endswith(dbsize, "K") ? "small-" : ""
    goldurl = "$MIRROR/public-queries/en-gold-standard-public/$(prefix)laion2B-en-public-gold-standard-$dbsize.h5"
    gfile = download_data(goldurl)
    
    res = evalresults(glob(joinpath(outdir, "*", "result-k=$k-*.h5")), gfile, k)
    CSV.write("results-$k-$dbsize.csv", res)
end
