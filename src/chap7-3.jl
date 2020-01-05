@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
end;

include("utils.jl")

df = CSV.read("/host/data/data-aircon.txt")

@gen function binominal_model(xs::Vector{Float64})
    b1 = @trace(uniform(-10, 10), :b1)
    b2 = @trace(uniform(-10, 10), :b2)
    x0 = @trace(uniform(0, 30), :x0)
    sigma = @trace(uniform(0, 1), :sigma)

    for (i, x) in enumerate(xs)
        mu = b1 + b2 * (x - x0) ^ 2
        @trace(normal(mu, sigma), (:y, i))
    end
end;


function do_inference(model, xs, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (xs, ), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:b1, :b2, :x0, :sigma)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(binominal_model, df.X, df.Y, 200000);
@show trace[:b1]
@show trace[:b2]
@show trace[:x0]
@show trace[:sigma]

df.Y_pred = trace[:b1] .+ (df.X .- trace[:x0]) .^ 2 * trace[:b2];
plt = lineplot(df.Y, df.Y);
println(scatterplot!(plt, df.Y, df.Y_pred));

plt = scatterplot(df.X, df.Y_pred);
println(scatterplot!(plt, df.X, df.Y))

