using Statistics
using Gen

function rmse(x, y)
    sqrt(sum((x .- y).^2)/length(x))
end

function inv_logit(x::Float64)
    1.0 / (1 + exp(-x))
end

function make_observation(ys)
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end
    return observations
end;

function predict(model, new_xs, n, trace, param_addrs)
    constraints = Gen.choicemap()
    for addr in param_addrs
        constraints[addr] = trace[addr]
    end
    (new_trace, _) = Gen.generate(model, new_xs, constraints)
    ys = [new_trace[(:y, i)] for i = 1:n]
    return ys
end