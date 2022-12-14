struct ProgramCausalState <: AbstractProgramState 
    progdgp::ProgramDGP

    μ::Float64
    τ::Float64
    σ::Float64

    programid::Int64
end

ProgramCausalState(μ::Float64, τ::Float64, σ::Float64, id::Int64) = ProgramCausalState(μ, τ, σ, id, nothing) 

function Base.rand(rng::Random.AbstractRNG, pd::ProgramDGP)
    μ_study = pd.μ + Base.rand(rng, Distributions.Normal(0, pd.η_μ))
    τ_study = pd.τ + Base.rand(rng, Distributions.Normal(0, pd.η_τ))

    return ProgramCausalState(pd, μ_study, τ_study, pd.σ, pd.programid)
end

dgp(ps::ProgramCausalState) = ps.progdgp

expectedutility(m::ExponentialUtilityModel, pcs::ProgramCausalState, a::AbstractFundingAction) = expectedutility(m, pcs.μ + (implements(a, pcs) ? pcs.τ : 0), pcs.σ)

getprogramid(ps::ProgramCausalState) = ps.programid

function Base.rand(rng::Random.AbstractRNG, ps::ProgramCausalState, samplesize::Int64 = 50) 
    y_control = Base.rand(rng, Normal(ps.μ, ps.σ), samplesize) 
    y_treated = Base.rand(rng, Normal(ps.μ + ps.τ, ps.σ), samplesize) 

    return ProgramEvalObservation(StudyDataset(y_control, y_treated), getprogramid(ps))
end

function logpdf(ps::ProgramCausalState, ds::StudyDataset) 
    controldist = Distributions.Normal(ps.μ, ps.σ) 
    treateddist = Distributions.Normal(ps.μ + ps.τ, ps.σ) 

    Σcontrollogpdf = reduce(ds.y_control, init = 0) do total, y
        total + log(pdf(controldist, y))
    end

    Σtreatedlogpdf = reduce(ds.y_treated, init = 0) do total, y
        total + log(pdf(treateddist, y))
    end

    return Σcontrollogpdf + Σtreatedlogpdf
end

logpdf(ps::ProgramCausalState, progobs::AbstractProgramEvalObservation) = logpdf(ps, progobs.d)

Distributions.pdf(ps::ProgramCausalState, ds::StudyDataset) = exp(logpdf(ps, ds))
Distributions.pdf(ps::ProgramCausalState, progobs::AbstractProgramEvalObservation) = exp(logpdf(ps, progobs))

Base.show(io::IO, ps::ProgramCausalState) = Printf.@printf(io, "ProgramCausalState(μ = %.2f, τ = %.2f, σ =  %.2f)", ps.μ, ps.τ, ps.σ)  

struct ProgramCausalStateDistribution
    pdgp::ProgramDGP
    prev::Union{Nothing, ProgramCausalState}
end

transition(pcs::ProgramCausalState, a::AbstractFundingAction) = ProgramCausalStateDistribution(dgp(pcs), pcs)

Base.rand(rng::Random.AbstractRNG, pcsd::ProgramCausalStateDistribution) = Base.rand(rng, pcsd.pdgp) #, pcsd.prev)

struct CausalState <: AbstractState 
    dgp::DGP
    programstates::Vector{ProgramCausalState}
    prev::Union{CausalState, Nothing}

    CausalState(dgp::DGP, ps::Vector{ProgramCausalState}, prev::Union{CausalState, Nothing} = nothing) = new(dgp, ps, prev)
end

CausalState(progstates::Vector{ProgramCausalState}) = CausalState(DGP([dgp(ps) for ps in progstates]), progstates)

dgp(s::CausalState) = s.dgp

function Base.rand(rng::Random.AbstractRNG, dgp::DGP, prev::Union{CausalState, Nothing} = nothing)
    return CausalState(dgp, [Base.rand(rng, pdgp) for pdgp in dgp.programdgps], prev)
end

Base.rand(rng::Random.AbstractRNG, s::CausalState, samplesize::Int64 = 50) = EvalObservation(Dict(getprogramid(ps) => Base.rand(rng, ps, samplesize) for ps in s.programstates))

Base.iterate(s::CausalState) = Base.iterate(s.programstates)
Base.iterate(s::CausalState, n) = Base.iterate(s.programstates, n)
Base.length(s::CausalState) = length(s.programstates)

numprograms(s::CausalState) = length(s.programstates)

getprogramstate(s::CausalState, id) = s.programstates[id]

expectedutility(m::ExponentialUtilityModel, s::CausalState, a::AbstractFundingAction) = StatsBase.sum([expectedutility(m, progstate, a) for progstate in s.programstates])

expectedutility(m::ExponentialUtilityModel, pc::ParticleFilters.ParticleCollection{CausalState}, a::AbstractFundingAction) = StatsBase.mean([expectedutility(m, s, a) for s in ParticleFilters.particles(pc)])

struct CausalStateDistribution
    dgp::AbstractDGP
    prev::Union{Nothing, CausalState}
end

Base.rand(rng::Random.AbstractRNG, sd::CausalStateDistribution) = Base.rand(rng, sd.dgp, sd.prev) 

function transition(s::CausalState, a::AbstractFundingAction)
    return CausalStateDistribution(s.dgp, s)
end
