@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
    using DataFramesMeta
end;

include("utils.jl")

df = CSV.read("/host/data/data-salary-2.txt")

@gen function line_model(xs::Vector{Float64})
    a = @trace(uniform(-1000, 1000), :a)
    b = @trace(uniform(-100, 100), :b)
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

trace = do_inference(line_model, df.X, df.Y, 100000);
@show trace[:a]
@show trace[:b]
@show trace[:sigma]

df.Y_pred = predict(line_model, (df.X, ), length(df.X), trace, [:a, :b, :sigma]);
plt = lineplot(df.Y, df.Y);
for i in 1:4
    dff = @where(df, :KID .== i)
    scatterplot!(plt, dff.Y, dff.Y_pred);
end
println(plt)



