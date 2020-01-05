@time begin
    using CSV
    using Gen
    using UnicodePlots
    using DataFramesMeta
    using ProgressMeter
end;
include("utils.jl")

df = CSV.read("/host/data/data-attendance-3.txt")


@gen function line_model(as::Vector{Int64}, scores::Vector{Float64})
    b1 = @trace(uniform(-10, 10), :b1)
    b2 = @trace(uniform(-10, 10), :b2)
    b3 = @trace(uniform(-10, 10), :b3)

    for (i, a) in enumerate(as)
        q = exp(b1 + b2 * as[i] + b3 * scores[i])
        @trace(poisson(q), (:y, i))
    end
end;


function do_inference(model, as, scores, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (as, scores), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:b1, :b2, :b3)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(line_model, df.A, df.Score / 200, df.M, 1000000);
@show trace[:b1]
@show trace[:b2]
@show trace[:b3]

df.q = exp.(trace[:b1] .+ df.A * trace[:b2] .+ df.Score / 200 * trace[:b3]);
df.M_pred = poisson.(df.q)
dfA1 = @where(df, :A .> 0);
dfA0 = @where(df, :A .== 0);

plt = lineplot(df.M, df.M);
scatterplot!(plt, dfA1.M, dfA1.M_pred);
println(scatterplot!(plt, dfA0.M, dfA0.M_pred));



