
struct StudyDataset
    y_control::Vector{Float64}
    y_treated::Vector{Float64}
end

function StudyDataset(n_control::Int64, n_treated::Int64)
    StudyDataset(Vector{Float64}(undef, n_control), Vector{Float64}(undef, n_treated))
end

function StudyDataset(n::Int64)
    StudyDataset(n, n)
end

function Base.show(io::IO, dataset::StudyDataset) 
    Printf.@printf(io, "StudyDataset(sample means = (%.2f, %.2f), sample SD = (%.2f, %.2f))", mean(dataset.y_control), mean(dataset.y_treated), std(dataset.y_control; corrected = false), std(dataset.y_treated; corrected = false))  
end

const StudyHistory = Vector{StudyDataset}

struct Hyperparam
    mu_sd::Float64 
    tau_mean::Float64
    tau_sd::Float64 
    sigma_sd::Float64 
    eta_sd::Vector{Float64}

    Hyperparam(; mu_sd, tau_mean, tau_sd, sigma_sd, eta_sd) = new(mu_sd, tau_mean, tau_sd, sigma_sd, eta_sd)
end

@model function sim_model(hyperparam::Hyperparam, datasets = missing; n_sim_study = 0, n_sim_obs = 0)
    if datasets === missing 
        n_study = n_sim_study 
        datasets = [StudyDataset(n_sim_obs) for i in 1:n_sim_study]
    else 
        n_study = length(datasets) 
    end

    μ_toplevel ~ Normal(0, hyperparam.mu_sd)
    τ_toplevel ~ Normal(hyperparam.tau_mean, hyperparam.tau_sd)
    σ_toplevel ~ truncated(Normal(0, hyperparam.sigma_sd), 0, Inf)
    η_toplevel ~ arraydist([truncated(Normal(0, hyperparam.eta_sd[i]), 0, Inf) for i in 1:2])

    μ_study ~ filldist(Normal(μ_toplevel, η_toplevel[1]), n_study)
    τ_study ~ filldist(Normal(τ_toplevel, η_toplevel[2]), n_study)
   
    for ds_index in 1:n_study
        for i in eachindex(datasets[ds_index].y_control) 
            datasets[ds_index].y_control[i] ~ Normal(μ_study[ds_index], σ_toplevel)
        end

        for i in eachindex(datasets[ds_index].y_treated) 
            datasets[ds_index].y_treated[i] ~ Normal(μ_study[ds_index] + τ_study[ds_index], σ_toplevel)
        end
    end

    return (datasets = datasets, μ_toplevel = μ_toplevel, μ_study = μ_study)
end

struct TuringModel <: AbstractBayesianModel 
    hyperparam::Hyperparam
    iter
    chains

    TuringModel(hyperparam::Hyperparam, iter = 500, chains = 4) = new(hyperparam, iter, chains)
end

function sample(m::TuringModel, datasets::Vector{StudyDataset})
    @pipe sim_model(m.hyperparam, datasets) |>
        Turing.sample(_, Turing.NUTS(), Turing.MCMCThreads(), m.iter, m.chains) |> 
        DataFrame(_) |>
        select(_, :μ_toplevel, :τ_toplevel, :σ_toplevel, r"η_toplevel", r"μ_study", r"τ_study") 
end 

#=
struct StanModel <: AbstractBayesianModel
    hyperparam::Hypergeometric
    model::StanSample.StampleModel
end

function StanModel(hyperparam::Hypergeometric)
    StanModel(hyperparam, StanSample.SampleModel("funding", "../stan/sim_model.stan", num_threads = 4, ))
end
=#