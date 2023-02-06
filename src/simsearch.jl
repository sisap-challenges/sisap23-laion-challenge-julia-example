using SimilaritySearch, JLD2
using Downloads: download


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

"""
    run_search(dist, kind, dfile, qfile, resfile)

Run search on different data and queries file using the given distance function.

The output results are stored in `resfile` (only `knns` and `dists` are mandatory fields)
"""
function run_search(dist::SemiMetric, k::Integer, db::AbstractDatabase, queries::AbstractDatabase, resfile::AbstractString)
    callbacks = SearchGraphCallbacks(ParetoRecall())
    verbose = true
    buildtime = @elapsed G = index!(SearchGraph(; db, dist, verbose); callbacks)
    opttime = @elapsed optimize!(G, MinRecall(0.9))
    searchtime = @elapsed knns, dists = searchbatch(G, queries, k)
    jldsave(resfile; knns, dists, buildtime, opttime, searchtime)
end

function main(kind, dist=SqL2Distance(), k=30)
    url_ = "http://ingeotec.mx/~sadit/SISAP23-Challenge"

    queriesurl = "$url_/$kind/en-queries/public-queries-10k-$kind.h5"
    dataseturl = "$url_/$kind/en-bundles/laion2B-en-$kind-n=100K.h5"
    goldurl = "$url_/public-queries/en-gold-standard-public/small-laion2B-en-public-gold-standard-100K.h5"

    qfile = download_data(queriesurl)
    dfile = download_data(dataseturl)
    gfile = download_data(goldurl)

    resfile = "knn-results-$k-" * basename(dfile)
    # loading or computing knns
    if !isfile(resfile)
        queries = StrideMatrixDatabase(jldopen(f->f[kind], qfile))
        db = StrideMatrixDatabase(jldopen(f->f[kind], dfile))
        run_search(dist, k, db, queries, resfile)
    end

    knns, buildtime, opttime, searchtime = jldopen(resfile) do f
        f["knns"], f["buildtime"], f["opttime"], f["searchtime"]
    end

    # loading gold standard for this dataset
    gold_knns = jldopen(f->f["knns"], gfile)
    recall = macrorecall(gold_knns[1:k, :], knns)
    println("""

    ===== $resfile =====
    buildtime: $buildtime
    opttime: $opttime
    searchtime: $searchtime
    recall: $recall
    ====================
    """)
end

main("pca32")
main("pca96")
main("hamming", BinaryHammingDistance())
