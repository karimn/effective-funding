begin
    using FundingPOMDPs
    using MCTS, POMDPs, D3Trees
    using DataFrames, DataFramesMeta
    using StatsBase, Gadfly, Pipe, Serialization
    using ParticleFilters
    using Distributions
    
    import SplitApplyCombine
end

include("diag_util.jl")

Gadfly.set_default_plot_size(50cm, 40cm)

file_suffix = "_500"

util_model = ExponentialUtilityModel(0.25)
#util_model = RiskNeutralUtilityModel()

begin
    all_sim_data = deserialize("temp-data/sim$(file_suffix).jls")
    greedy_sim_data = @subset(all_sim_data, :plan_type .== "none")
    pftdpw_sim_data = @subset(all_sim_data, :plan_type .== "pftdpw")
    random_sim_data = @subset(all_sim_data, :plan_type .== "random")
    freq_sim_data = @subset(all_sim_data, :plan_type .== "freq")
end

do_nothing_reward = begin 
    do_nothing = ImplementEvalAction()

    @pipe greedy_sim_data |>
        dropmissing(_, :state) |>
        @select(
            _, 
            :actual_reward = [expectedutility.(Ref(util_model), states[Not(end)], Ref(do_nothing)) for states in :state],
            :actual_ex_ante_reward = [expectedutility.(Ref(util_model), dgp.(states[Not(end)]), Ref(do_nothing)) for states in :state],
            :plan_type = "no eval"
        )  
end

do_best_reward = @pipe greedy_sim_data |>
    dropmissing(_, :state) |> 
    @select(
        _,
        :actual_reward = map(get_program_reward, :state),
        :actual_ex_ante_reward = map(s -> get_program_reward(s, eval_getter = dgp), :state),
        :plan_type = "best"
    ) 

accum_rewards = true 
maxstep = 15

util_diff = @pipe all_sim_data |> 
    groupby(_, :plan_type) |>
    @select(
        _,
        :plan_type,
        :ex_post_reward_diff = calculate_util_diff(:actual_reward, do_nothing_reward.actual_reward, accum = false, maxstep = maxstep),
        :accum_ex_post_reward_diff = calculate_util_diff(:actual_reward, do_nothing_reward.actual_reward, accum = true, maxstep = maxstep),
        :ex_ante_reward_diff = calculate_util_diff(:actual_ex_ante_reward, do_nothing_reward.actual_ex_ante_reward, accum = false, maxstep = maxstep),
        :accum_ex_ante_reward_diff = calculate_util_diff(:actual_ex_ante_reward, do_nothing_reward.actual_ex_ante_reward, accum = true, maxstep = maxstep),
        :step = repeat([collect(1:maxstep)], length(:plan_type))
    ) |>  
    transform(_, eachindex => :sim) |> 
    flatten(_, Not([:sim, :plan_type])) 

util_diff_summ = @pipe util_diff |> 
    

vg_util_diff_summ = @pipe [pftdpw_sim_data.actual_reward, random_sim_data.actual_reward, freq_sim_data.actual_reward, do_ex_post_best_reward] |> 
    map(r -> filter(x -> length(x) >= maxstep, r), _) |> 
    (calculate_util_diff_summ ∘ calculate_util_diff).(_, Ref(greedy_sim_data.actual_reward); accum = accum_rewards, maxstep = maxstep) |> 
    vcat(_...; source = :algo => ["planned", "random", "freq", "ex post best"])

vmax_util_diff_summ = @pipe [pftdpw_sim_data.actual_reward, random_sim_data.actual_reward, freq_sim_data.actual_reward, greedy_sim_data.actual_reward] |> 
    map(r -> filter(x -> length(x) >= maxstep, r), _) |> 
    (calculate_util_diff_summ ∘ calculate_util_diff).(_, Ref(do_ex_post_best_reward); accum = accum_rewards, maxstep = maxstep) |> 
    vcat(_...; source = :algo => ["planned", "random", "freq", "greedy"])

util_diff_summ = @pipe [greedy_sim_data.actual_reward, pftdpw_sim_data.actual_reward, random_sim_data.actual_reward, freq_sim_data.actual_reward, do_ex_post_best_reward] |>
    map(r -> filter(x -> length(x) >= maxstep, r), _) |> 
    (calculate_util_diff_summ ∘ calculate_util_diff).(_, Ref(do_nothing_reward); accum = accum_rewards, maxstep = maxstep) |>
    vcat(_...; source = :algo => ["greedy", "planned", "random", "freq", "ex post best"])  
#    @rsubset(_, :algo in ("greedy", "planned", "ex post best", "random"))

@pipe vstack(
    plot(
        util_diff_summ, x = :step, 
        layer(y = :mean, color = :algo, linestyle = [:dash], Geom.point, Geom.line),
        layer(y = :med, ymin = :lb, ymax = :ub,  color = :algo,alpha = [0.75], Geom.line, Geom.ribbon),
        layer(yintercept = [0.0], Geom.hline(style = :dot, color = "grey")),
        Scale.x_discrete, Guide.yticks(ticks = -0.1:0.1:3) 
    ),

    plot(
        vg_util_diff_summ, x = :step, color = :algo, 
        layer(y = :mean, linestyle = [:dash], Geom.point, Geom.line),
        layer(y = :med, ymin = :lb, ymax = :ub, alpha = [0.75], Geom.line, Geom.ribbon),
        layer(yintercept = [0.0], Geom.hline(style = :dot, color = "grey")),
        Scale.x_discrete, Guide.yticks(ticks = -0.3:0.1:2) 
    )
) 

plot(
    vmax_util_diff_summ, x = :step, color = :algo, 
    layer(y = :mean, linestyle = [:dash], Geom.point, Geom.line),
    layer(y = :med, ymin = :lb, ymax = :ub, alpha = [0.75], Geom.line, Geom.ribbon),
    layer(yintercept = [0.0], Geom.hline(style = :dot, color = "grey")),
    Scale.x_discrete, Guide.yticks(ticks = -0.3:0.1:2) 
)

obs_reward = @pipe pairs((greedy = greedy_sim_data.actual_reward, planned = pftdpw_sim_data.actual_reward, random = random_sim_data.actual_reward, freq = freq_sim_data.actual_reward)) |>
    DataFrame(_) |>
    insertcols!(_, :sim => 1:nrow(_)) |>
    DataFrames.flatten(_, [:greedy, :planned, :random, :freq]) |>
    groupby(_, :sim) |>
    transform!(_, eachindex => :step) |>
    stack(_, [:greedy, :planned, :random, :freq], value_name = :reward) |>
    groupby(_, Cols(:sim, :variable)) |>
    @transform!(_, :cumul_reward = cumsum(:reward))

nsteps = maximum(obs_reward.step)

b_data_greedy = @pipe get_beliefs_data.(greedy_sim_data.belief) |> 
    vcat(_..., source = :sim)

nprograms = maximum(b_data_greedy.programid)

b_data_planned = @pipe get_beliefs_data.(pftdpw_sim_data.belief) |> 
    vcat(_..., source = :sim)

b_data_random = @pipe get_beliefs_data.(random_sim_data.belief) |> 
    vcat(_..., source = :sim)

last_b_data_greedy = @pipe get_beliefs_data.(greedy_sim_data.belief; forecast = false) |> 
    vcat(_..., source = :sim)

last_b_data_planned = @pipe get_beliefs_data.(pftdpw_sim_data.belief; forecast = false) |> 
    vcat(_..., source = :sim)

last_b_data_random = @pipe get_beliefs_data.(random_sim_data.belief; forecast = false) |> 
    vcat(_..., source = :sim)

summarize_b_data(b_data, probs = [0.1, 0.5, 0.9]) = @pipe b_data |>
    groupby(_, Cols(:sim, :programid, :step)) |>
    vcat(
        @combine(_, :per = string.("μ_per_", probs), :quant = quantile(:μ, probs)),
        @combine(_, :per = string.("τ_per_", probs), :quant = quantile(:τ, probs)) 
    ) |> 
    unstack(_, :per, :quant) #; renamecols = c -> "per_$c") 

b_data_greedy_summ = summarize_b_data(b_data_greedy) 
b_data_planned_summ = summarize_b_data(b_data_planned) 
b_data_random_summ = summarize_b_data(b_data_random) 
last_b_data_greedy_summ = summarize_b_data(last_b_data_greedy) 
last_b_data_planned_summ = summarize_b_data(last_b_data_planned)  
last_b_data_random_summ = summarize_b_data(last_b_data_random)  

s_data = @pipe get_states_data.(greedy_sim_data.state) |>
    vcat(_..., source = :sim)

actlist = @pipe SelectProgramSubsetActionSetFactory(nprograms, 1) |> FundingPOMDPs.actions(_).actions

all_rewards = @pipe get_rewards_data.(greedy_sim_data.state, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "actual")

greedy_all_expected_rewards = @pipe get_rewards_data.(greedy_sim_data.belief, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "expected greedy")

all_expected_rewards = @pipe get_rewards_data.(pftdpw_sim_data.belief, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "expected planned")

planned_obs_act = get_actions_data(pftdpw_sim_data.action)
random_obs_act = get_actions_data(random_sim_data.action)

pdgps_data = @pipe pftdpw_sim_data.state |>
    get_dgp_data.(_[1]) |>
    [ @transform!(d[2], :sim = d[1]) for d in enumerate(_)] |>
    vcat(_...)

begin
    sim = 3 
    
    obs_reward_plot = @pipe obs_reward |>
        @rsubset(_, :sim in sim) |>
        #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
        plot(_, x = :step, y = :reward, xgroup = :sim, color = :variable, Geom.subplot_grid(Geom.point, Geom.line), Scale.x_discrete)

    obs_cumul_reward_plot = @pipe obs_reward |>
        @rsubset(_, :sim in sim) |>
        #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
        plot(_, x = :step, y = :cumul_reward, xgroup = :sim, color = :variable, Geom.subplot_grid(Geom.point, Geom.line), Scale.x_discrete)

    reward_plot = @pipe vcat(all_rewards, all_expected_rewards, greedy_all_expected_rewards) |>
        @subset(_, :sim .== sim, :reward_type .== "actual") |>
        plot(
            _, x = :step, y = :reward, color = :actprog, linestyle = :reward_type, 
            Geom.line, Scale.color_discrete, Scale.x_discrete,
            Guide.title("True Rewards"), 
        )

    cumul_reward_plot =  
        @pipe vcat(all_rewards, all_expected_rewards, greedy_all_expected_rewards) |>
            @subset(_, :sim .== sim, :reward_type .== "actual") |>
            plot(
                _, x = :step, y = :cumul_reward, color = :actprog, linestyle = :reward_type,
                Geom.line, Scale.color_discrete, Scale.x_discrete,
                Guide.title("Cumulative Reward")
            )

    planned_act_plot = @pipe planned_obs_act |>
        @rsubset(_, :sim in sim) |>
        @rtransform!(_, :pid = :action_type == "implement_programs" ? :pid - 0.02 : :pid + 0.02) |>
        plot(
            _, x = :step, y = :pid, color = :action_type, 
            Geom.step, style(line_width = 1mm), Scale.x_discrete, Coord.cartesian(ymin = 0, ymax = nprograms),
            Guide.title("Planned Actions")
        )

    random_act_plot = @pipe random_obs_act |>
        @rsubset(_, :sim in sim) |>
        @rtransform!(_, :pid = :action_type == "implement_programs" ? :pid - 0.02 : :pid + 0.02) |>
        plot(
            _, x = :step, y = :pid, color = :action_type, 
            Geom.step, style(line_width = 1mm), Scale.x_discrete, Coord.cartesian(ymin = 0, ymax = nprograms),
            Guide.title("Random Actions")
        )

    belief_plot_data = @pipe planned_obs_act |>
        @subset(_, :sim .== sim, :action_type .== "eval_programs") |>
        @transform!(_, :step = :step .+ 1) |>
        vcat(
            @subset(b_data_planned_summ, :sim .== sim, :step .== 1),
            rightjoin(b_data_planned_summ, _, on = [:sim, :step, (:programid => :pid)]);
            cols = :union
        )
        
    μ_belief_plot = @pipe belief_plot_data |> 
        plot(
            _, x = :step, ymin = "μ_per_0.1", ymax = "μ_per_0.9", color = :programid,
            layer(alpha = [0.2], Geom.ribbon, Stat.dodge),
            layer(Geom.yerrorbar, Stat.dodge, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            layer(@subset(s_data, :sim .== sim), y = :μ, Geom.line, Stat.dodge),
            Scale.discrete_color, Scale.x_discrete(levels = 1:(nsteps + 1)), Guide.ylabel("μ"),
            Guide.title("Predicted States")
        )

    τ_belief_plot = @pipe belief_plot_data |> 
        plot(
            _, x = :step, ymin = "τ_per_0.1", ymax = "τ_per_0.9", color = :programid,
            layer(alpha = [0.2], Geom.ribbon, Stat.dodge),
            layer(Geom.yerrorbar, Stat.dodge, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            layer(@subset(s_data, :sim .== sim), y = :τ, Geom.line, Stat.dodge),
            Scale.discrete_color, Scale.x_discrete(levels = 1:(nsteps + 1)), Guide.ylabel("τ"),
            Guide.title("Predicted States")
        )

    last_belief_plot_data = @pipe planned_obs_act |>
        @subset(_, :sim .== sim, :action_type .== "eval_programs") |>
        @transform!(_, :step = :step .+ 1) |>
        vcat(
            @subset(last_b_data_planned_summ, :sim .== sim, :step .== 1),
            rightjoin(last_b_data_planned_summ, _, on = [:sim, :step, (:programid => :pid)]);
            cols = :union
        ) |>
        @transform!(_, :step = :step .- 1)
        
    μ_last_belief_plot = @pipe last_belief_plot_data |> 
        plot(
            _, x = :step, y = "μ_per_0.5", ymin = "μ_per_0.1", ymax = "μ_per_0.9", color = :programid,
            layer(Geom.line),
            layer(alpha = [0.8], Geom.ribbon),
            layer(Geom.yerrorbar, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            Scale.discrete_color, Scale.x_discrete(levels = 0:nsteps), Guide.ylabel("μ"),
            Guide.title("Last States")
        )
        
    τ_last_belief_plot = @pipe last_belief_plot_data |>
        plot(
            _, x = :step, y = "τ_per_0.5", ymin = "τ_per_0.1", ymax = "τ_per_0.9", color = :programid,
            layer(Geom.line),
            layer(alpha = [0.8], Geom.ribbon),
            layer(Geom.yerrorbar, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            Scale.discrete_color, Scale.x_discrete(levels = 0:nsteps), Guide.ylabel("τ"),
            Guide.title("Last States")
        )

    τ_density_data = @pipe pdgps_data |>
        @subset(_, :sim .== sim) |>
        @rselect!(_, :programid, :τ = rand(Distributions.Normal(:τ, :η_τ), 10000)) |>
        DataFrames.flatten(_, :τ) 

    #=
    first_belief_hist = @pipe pairs((planned = b_data_planned, greedy = b_data_greedy)) |>
        [@transform(v, :policy = k) for (k, v) in _] |>
        vcat(_...) |>
        @subset(_, :sim .== sim, :step .== 1) |>
        plot(_, x = :τ, xgroup = :programid, alpha = [0.5], color = :policy, Geom.subplot_grid(Geom.histogram(position = :identity, density = true), Scale.x_continuous(minvalue = -5, maxvalue = 5)))

    first_belief = @pipe b_data_planned_summ |>
        @subset(_, :sim .== sim, :step .== 1) |>
        #plot(_, x = :τ, xgroup = :programid, alpha = [0.5], Geom.subplot_grid(Geom.histogram(density = true), Scale.x_continuous(minvalue = -5, maxvalue = 5)))
        plot(
            layer(_, x = :programid, y = "per_0.5", ymin = "per_0.1", ymax = "per_0.9", Geom.errorbar, Geom.point, style(errorbar_cap_length=0mm, line_width = 0.5mm)),
            layer(τ_density_data, x = :programid, y = :τ, alpha = [0.1], Geom.violin),
            Scale.x_discrete
        )
    =#

    title(gridstack([obs_reward_plot     obs_cumul_reward_plot; 
                     reward_plot         cumul_reward_plot; 
                     planned_act_plot    random_act_plot;
                     μ_belief_plot       τ_belief_plot; 
                     μ_last_belief_plot  τ_last_belief_plot]), "Simulation $sim")
end

#@pipe map(0:0.1:2) do α
@pipe map([0.5]) do α
    um = ExponentialUtilityModel(α)

    @pipe DataFrame(α = α, x = -1.5:0.1:1.5) |>
        @transform!(_, :u = utility.(Ref(um), :x))
end |>
    vcat(_...) |>
    plot(_, x = :x , y = :u, color = :α, Geom.line, Coord.cartesian(fixed = true))


@pipe DataFrame(x = 0:20) |> 
    @rtransform!(_, :y = 10 * (:x^(1/10))) |> 
    plot(_, x = :x, y = :y, Geom.line, Geom.point, Guide.yticks(ticks = 0.0:0.5:6))

pb2 = programbeliefs(pftdpw_sim_data.belief[4][12])[2]
expectedutility(util_model, pb2, false)
expectedutility(util_model, pb2, true)

pb2d_data = @pipe pb2 |> 
    state_samples(_) |>
    particles(_) |> 
    [DataFrame(implemented = a, meanlevel = expectedlevel.(_, a), meanutil = expectedutility.(Ref(util_model), _, a)) for a in (false, true) ] |> 
    vcat(_...)  

@pipe pb2d_data |> 
    groupby(_, :implemented) |> 
    @combine(_, :lvl_std = std(:meanlevel), :eu_std = std(:meanutil))

plot(pb2d_data, x = :meanlevel, y = :meanutil, color = :implemented, Geom.density2d)
plot(pb2d_data, x = :meanutil, color = :implemented, Geom.density)
plot(pb2d_data, x = :meanlevel, color = :implemented, Geom.density)

function MCTS.node_tag(b::CausalStateParticleBelief) 
    "[Belief]"
end

function MCTS.node_tag(a::SeparateImplementEvalAction) 
    eval_text = string([p for p in a.eval_programs])
    implement_text = isempty(a.implement_programs) ? "[]" : string([p for p in a.implement_programs]) 

    return "{i=$implement_text, e=$eval_text}"
end

inchrome(D3Tree(pftdpw_sim_data.tree[1][1], init_expand = 1))
inchrome(D3Tree(pftdpw_sim_data.tree[1][2], init_expand = 1))
inchrome(D3Tree(pftdpw_sim_data.tree[1][3], init_expand = 1))

@pipe obs_reward |>
        @rsubset(_, :sim in 3) |>
        #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
        plot(_, x = :step, y = :cumul_reward, xgroup = :sim, color = :variable, Geom.subplot_grid(Geom.point, Geom.line), Scale.x_discrete)


wise_bayes_model = TuringModel(dgp_priors; iter = NUM_TURING_MODEL_ITER)

#=
greedy_sim_data.action[2]
freq_sim_data.action[2]
greedy_sim_data.belief[2][1].progbeliefs[5]
freq_sim_data.belief[2][1].progbeliefs[5]
data(greedy_sim_data.belief[2][1].progbeliefs[5])
data(freq_sim_data.belief[2][1].progbeliefs[5])
last_state_samples(greedy_sim_data.belief[3][1].progbeliefs[1])
last_state_samples(freq_sim_data.belief[3][1].progbeliefs[1])
greedy_sim_data.state[3][2].programstates
expectedutility(util_model, greedy_sim_data.belief[3][1].progbeliefs[1], false)
expectedutility(util_model, freq_sim_data.belief[3][1].progbeliefs[1], true)
dgp(greedy_sim_data.state[2][1].programstates[5])
dgp(freq_sim_data.state[2][1].programstates[5])

bayes_model = TuringModel(inference_priors; iter = NUM_TURING_MODEL_ITER)

s0 = FundingPOMDPs.sample(bayes_model, data(greedy_sim_data.belief[2][1].progbeliefs[5]), progress = true)

s1 = FundingPOMDPs.sample(wise_bayes_model, data(greedy_sim_data.belief[2][1].progbeliefs[5]), progress = true)

naive_bayes_model = TuringModel(inference_priors; iter = NUM_TURING_MODEL_ITER, multilevel = false)

s2 = FundingPOMDPs.sample(naive_bayes_model, data(greedy_sim_data.belief[2][1].progbeliefs[5]), progress = true)

b = ProgramBelief(wise_bayes_model, data(greedy_sim_data.belief[2][1].progbeliefs[5]), 5, Random.GLOBAL_RNG)
=#

fb = freq_sim_data.belief[2][1] 
b = pftdpw_sim_data.belief[2][1] 
b.progbeliefs
fb.progbeliefs
expectedutility(util_model, b, ImplementEvalAction([4]))
expectedutility(util_model, fb, ImplementEvalAction([4]))
expectedutility.(Ref(util_model), Ref(fb), actlist)

last_state_samples(b.progbeliefs[4])
s1 = FundingPOMDPs.sample(wise_bayes_model, data(b.progbeliefs[4]), progress = true)