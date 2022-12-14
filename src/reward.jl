struct ExponentialUtilityModel <: AbstractRewardModel
    α::Float64
end

utility(m::ExponentialUtilityModel, outcome) = 1 - exp(- m.α * outcome)
expectedutility(m::ExponentialUtilityModel, μ::Float64, σ::Float64) = 1 - exp(- m.α * μ + m.α^2 * σ^2 / 2)
expectedutility(m::ExponentialUtilityModel, μ::AbstractVector, σ::AbstractVector) = 1 .- exp.(- m.α .* μ + m.α^2 .* σ.^2 / 2)
