using SimilaritySearch, JLD2, CSV
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

function run_searchgraph(dist::SemiMetric, k::Integer, db::AbstractDatabase, queries::AbstractDatabase, meta, resfile::AbstractString)
    algo = "SearchGraph"
    opt = MinRecall(0.9)
    callbacks = SearchGraphCallbacks(opt)
    verbose = false
    buildtime = @elapsed G = index!(SearchGraph(; db, dist, verbose); callbacks)
    buildtime += @elapsed optimize!(G, opt)
    querytime = @elapsed knns, dists = searchbatch(G, queries, k)
    jldsave(resfile; knns, dists, algo, buildtime, querytime, params="MinRecall 0.9", meta...)
end

function run_bruteforce(dist::SemiMetric, k::Integer, db::AbstractDatabase, queries::AbstractDatabase, meta, resfile::AbstractString)
    algo = "bruteforce"
    E = ExhaustiveSearch(; dist, db)
    buildtime = 0.0
    querytime = @elapsed knns, dists = searchbatch(E, queries, k)
    jldsave(resfile; knns, dists, algo, buildtime, querytime, params="", meta...)
end

MIRROR = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"

function main(kind, dbsize, k, dist=SqL2Distance(); resultsdir="results-$k")
    queriesurl = "$MIRROR/$kind/en-queries/public-queries-10k-$kind.h5"
    dataseturl = "$MIRROR/$kind/en-bundles/laion2B-en-$kind-n=$dbsize.h5"

    qfile = download_data(queriesurl)
    dfile = download_data(dataseturl)
    mkpath(resultsdir)
    @info "loading $qfile and $dfile"
    queries = StrideMatrixDatabase(jldopen(f->f[kind], qfile))
    db = StrideMatrixDatabase(jldopen(f->f[kind], dfile))

    # loading or computing knns
    resfile_bruteforce = joinpath(resultsdir, "searchgraph-k=$k-size=$dbsize-" * basename(dfile))
    resfile_searchgraph = joinpath(resultsdir, "bruteforce-k=$k-size=$dbsize-" * basename(dfile))
    meta = (size=dbsize, data=kind)
    run_bruteforce(dist, k, db, queries, meta, resfile_bruteforce)
    run_searchgraph(dist, k, db, queries, meta, resfile_searchgraph)
end

for dbsize in ("100K", "300K")
    k = 30
    resultsdir = "results-k=$k-size=$dbsize"
    main("hamming", dbsize, k, BinaryHammingDistance(); resultsdir)
    main("pca32", dbsize, k; resultsdir)
    main("pca96", dbsize, k; resultsdir)

    prefix = endswith(dbsize, "K") ? "small-" : ""
    goldurl = "$MIRROR/public-queries/en-gold-standard-public/$(prefix)laion2B-en-public-gold-standard-$dbsize.h5"
    gfile = download_data(goldurl)
    evalresults(resultsdir, gfile, k)
end
