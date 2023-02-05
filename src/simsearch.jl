using SimilaritySearch, JLD2, Parquet2, HDF5, FileIO, JSON
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

function main(kind, dist=SqL2Distance(), k=30)
    url_ = "http://ingeotec.mx/~sadit/SISAP23-Challenge"
    
    queriesurl = "$url_/$kind/en-queries/public-queries-10k-$kind.h5"
    dataseturl = "$url_/$kind/en-bundles/laion2B-en-$kind-n=100K.h5"
    goldurl = "$url_/public-queries/en-gold-standard-public/small-laion2B-en-public-gold-standard-100K.h5"

    qfile = download_data(queriesurl)
    dfile = download_data(dataseturl)
    gfile = download_data(goldurl)

    queries = StrideMatrixDatabase(load(qfile, kind))
    db = StrideMatrixDatabase(load(dfile, kind))

    resfile = "knn-results-$k-" * basename(dfile)
    # loading or computing knns
    if isfile(resfile)
        knns, buildtime, opttime, searchtime = load(resfile, "knns", "buildtime", "opttime", "searchtime")
    else
        callbacks = SearchGraphCallbacks(ParetoRecall())
        verbose = true
        buildtime = @elapsed G = index!(SearchGraph(; db, dist, verbose); callbacks)
        opttime = @elapsed optimize!(G, MinRecall(0.9))
        searchtime = @elapsed knns, dists = searchbatch(G, queries, k)
        jldsave(resfile; knns, dists, buildtime, opttime, searchtime)
    end

    # loading gold standard for this dataset
    gold_knns = load(gfile, "knns")
    @show resfile
    @show buildtime, opttime, searchtime, macrorecall(gold_knns[1:k, :], knns)
end

main("pca32")
main("pca96")
main("hamming", BinaryHammingDistance())
