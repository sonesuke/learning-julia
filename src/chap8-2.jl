@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
    using DataFramesMeta
end;

include("utils.jl")

df = CSV.read("/host/data/data-salary-2.txt")

@gen function line_model(xs::Vector{Float64}, kids::Vector{Int64})
    a = [@trace(uniform(-1000, 1000), (:a, i)) for i=1:4]
    b = [@trace(uniform(-100, 100), (:b, i)) for i=1:4]
    sigma = @trace(uniform(0, 100), :sigma)

    for (i, x) in enumerate(xs)
        @trace(normal(a[kids[i]] + b[kids[i]] * x, sigma), (:y, i))
    end
end;

function model_symbols()
    a = [(:a, i) for i = 1:4]
    b = [(:b, i) for i = 1:4]
    [a; b; :sigma]
end

function do_inference(model, xs, kids, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (xs, kids), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select((:a, 1), (:a, 2), (:a, 3), (:a, 4), (:b, 1), (:b, 2), (:b, 3), (:b, 4), :sigma)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(line_model, df.X, df.KID, df.Y, 100000);
@show trace[(:a, 1)]
@show trace[(:a, 2)]
@show trace[(:a, 3)]
@show trace[(:a, 4)]
@show trace[(:b, 1)]
@show trace[(:b, 2)]
@show trace[(:b, 3)]
@show trace[(:b, 4)]
@show trace[:sigma]

df.Y_pred = predict(line_model, (df.X, df.KID), length(df.X), trace, model_symbols());
plt = lineplot(df.Y, df.Y);
for i in 1:4
    dff = @where(df, :KID .== i)
    scatterplot!(plt, dff.Y, dff.Y_pred);
end
println(plt)



