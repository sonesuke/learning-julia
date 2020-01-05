@time begin
    using CSV
    using Gen
    using UnicodePlots
    using DataFramesMeta
    using ProgressMeter
end;
include("utils.jl")

df = CSV.read("/host/data/data-attendance-3.txt")

@gen function line_model(as::Vector{Int64}, scores::Vector{Float64}, weathers::Vector{Float64})
    b1 = @trace(uniform(-1, 1), :b1)
    b2 = @trace(uniform(-1, 1), :b2)
    b3 = @trace(uniform(-1, 1), :b3)
    b4 = @trace(uniform(-1, 1), :b4)

    for (i, a) in enumerate(as)
        q = inv_logit(b1 + b2 * as[i] + b3 * scores[i] + b4 * weathers[i])
        @trace(bernoulli(q), (:y, i))
    end
end;


function do_inference(model, as, scores, weathers, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (as, scores, weathers), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:b1, :b2, :b3, :b4)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;


function encode(label::String)
    label_map = Dict("A" => 0, "B" => 0.2, "C" => 1)
    return label_map[label]
end;


df.W = encode.(df.Weather)
trace = do_inference(line_model, df.A, df.Score / 200, df.W, df.Y, 100000);
@show trace[:b1]
@show trace[:b2]
@show trace[:b3]
@show trace[:b4]

df.q = inv_logit.(trace[:b1] .+ df.A * trace[:b2] .+ df.Score / 200 * trace[:b3] + df.W * trace[:b4]);
df.Y_pred = bernoulli.(df.q)
@by(df, [:Y, :Y_pred], sum_Y = length(:Y))


