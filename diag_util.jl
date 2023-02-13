function calculate_util_diff(planned_reward, baseline_reward; accum = false, maxstep = nothing)  
    if accum
        diff = map((p, n) -> cumsum(p) - cumsum(n), planned_reward, baseline_reward)  
    else
        diff = map((p, n) -> p - n, planned_reward, baseline_reward)  
    end

    if maxstep ≢ nothing
        diff = map(diff) do sim_diff
            sim_diff[1:maxstep]
        end
    end

    return diff
end

function calculate_util_diff_summ(util_diff)
    util_diff_mean = [mean(a) for a in SplitApplyCombine.invert(util_diff)]
    util_diff_quant = @pipe [quantile(skipmissing(a), [0.25, 0.5, 0.75]) for a in SplitApplyCombine.invert(util_diff)] |> 
        DataFrame(SplitApplyCombine.invert(_), [:lb, :med, :ub]) |>
        insertcols!(_, :step => 1:nrow(_), :mean => util_diff_mean)

    return util_diff_quant
end