using Distributions
using Gen

struct Binomial <: Gen.Distribution{Int} end

"""
    binomial(N, theta::Real)
Sample an `Int` from the binomial distribution with trial `N` and rate `theta`.
"""
const binomial = Binomial()

function Gen.logpdf(::Binomial, x::Integer, N::Integer, theta::Real)
    Distributions.logpdf(Distributions.Binomial(N, theta), x)
end


function Gen.logpdf_grad(::Binomial, x::Integer, N::Integer, theta::Real)
    error("Not implemented")
    (nothing, nothing)
end


function Gen.random(::Binomial, N::Integer, theta::Real)
    rand(Distributions.Binomial(N, theta))
end


(::Binomial)(N, theta) = random(Binomial(), N, theta)
Gen.is_discrete(::Binomial) = true
Gen.has_output_grad(::Binomial) = false
Gen.has_argument_grads(::Binomial) = (false, false)

export binomial
