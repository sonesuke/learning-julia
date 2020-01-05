@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
end;

include("utils.jl")

df = CSV.read("/host/data/data-salary.txt")

@gen function line_model(xs::Vector{Float64})
    a = @trace(uniform(-1000, 1000), :a)
    b = @trace(uniform(-1000, 1000), :b)
    sigma = @trace(uniform(0, 100), :sigma)

    for (i, x) in enumerate(xs)
        @trace(normal(a + b * x, sigma), (:y, i))
    end
end;


function do_inference(model, xs, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (xs, ), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:a, :b, :sigma)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(line_model, df.X, df.Y, 30000);
@show trace[:a]
@show trace[:b]
@show trace[:sigma]

@show "results"
df.Y_pred = trace[:a] .+ df.X * trace[:b];
plt = lineplot(df.Y, df.Y);
println(scatterplot!(plt, df.Y, df.Y_pred));



