@time begin
    using CSV
    using Gen
    using UnicodePlots
    using DataFramesMeta
    using ProgressMeter
end;

include("utils.jl")

df = CSV.read("/host/data/data-attendance-1.txt")

@gen function line_model(as::Vector{Int64}, scores::Vector{Float64})
    b1 = @trace(uniform(-1, 1), :b1)
    b2 = @trace(uniform(-1, 1), :b2)
    b3 = @trace(uniform(-1, 1), :b3)

    sigma = @trace(uniform(0, 1), :sigma)

    for (i, a) in enumerate(as)
        @trace(normal(b1 + b2 * as[i] + b3 * scores[i], sigma), (:y, i))
    end
end;

function do_inference(model, as, scores, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (as, scores), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:b1, :b2, :b3, :sigma)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(line_model, df.A, df.Score / 200.0, df.Y, 200000);
@show trace[:b1]
@show trace[:b2]
@show trace[:b3]
@show trace[:sigma]

df.Y_pred = trace[:b1] .+ df.A * trace[:b2] .+ df.Score / 200 * trace[:b3];
dfA1 = @where(df, :A .> 0);
dfA0 = @where(df, :A .== 0);

plt = lineplot(df.Y, df.Y);
scatterplot!(plt, dfA1.Y, dfA1.Y_pred);
println(scatterplot!(plt, dfA0.Y, dfA0.Y_pred));



