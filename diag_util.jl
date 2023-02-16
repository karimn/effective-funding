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

function get_program_reward(sim_states; eval_getter = identity) 
    implement_only_asf = SelectProgramSubsetActionSetFactory(FundingPOMDPs.numprograms(first(sim_states)), 1)

    fixed_action_reward = [expectedutility.(Ref(util_model), eval_getter.(sim_states[Not(end)]), Ref(a)) for a in FundingPOMDPs.actions(implement_only_asf)]

    argmax(sum, fixed_action_reward)
end

function summarize_util_diff(sim_data, compare_to, best_reward = nothing; ex_ante, maxstep, accum)
    @pipe sim_data |> 
        groupby(_, :plan_type) |> 
        getindex.(_, :, ex_ante ? :actual_ex_ante_reward : :actual_reward) |> 
        vcat(_, [best_reward]) |> 
        map(r -> filter(x -> length(x) >= maxstep, r), _) |> 
        (calculate_util_diff_summ ∘ calculate_util_diff).(_, Ref(compare_to); accum = accum, maxstep = maxstep) |>
        vcat(_...; source = :algo => ["greedy", "planned", "random", "freq", "ex post best"])  
end