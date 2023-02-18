function evalresults(resultsdir, gfile, k)
    gold_knns = jldopen(f->f["knns"], gfile)
    res = (data=[], size=[], algo=[], buildtime=[], querytime=[], params=[], recall=[])
    for resfile in readdir(resultsdir, join=true)
        reg = jldopen(resfile) do f
            knns = f["knns"]
            recall = macrorecall(gold_knns[1:k, :], knns)
            (f["data"], f["size"], f["algo"], f["buildtime"], f["querytime"], f["params"], recall)  # aligned with res
        end

        for i in eachindex(reg)
            push!(res[i], reg[i])
        end
    end

    resfile = basename(rstrip(resultsdir, '/')) * ".csv"
    CSV.write(resfile, res)
end