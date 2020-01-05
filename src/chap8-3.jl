@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
    using DataFramesMeta
    using MLBase
end;

include("utils.jl")

df = CSV.read("/host/data/data-salary-2.txt")

@gen function line_model(df)
    a0 = @trace(uniform(0, 1000), :a0)
    b0 = @trace(uniform(0, 100), :b0)
    sigma_a = @trace(uniform(0, 100), :sigma_a)
    sigma_b = @trace(uniform(0, 100), :sigma_b)
    ak = [@trace(normal(0, sigma_a), (:ak, i)) for i=1:4]
    bk = [@trace(normal(0, sigma_b), (:bk, i)) for i=1:4]
    a = [a0 + ak[i] for i=1:4]
    b = [b0 + bk[i] for i=1:4]
    sigma = @trace(uniform(0, 100), :sigma)

    for (i, x) in enumerate(df.X)
        @trace(normal(a[df.KID[i]] + b[df.KID[i]] * x, sigma), (:y, i))
    end
end;

function model_symbols()
    ak = [(:ak, i) for i = 1:4]
    bk = [(:bk, i) for i = 1:4]
    [:a0; :b0; :sigma_a; :sigma_b; ak; bk; :sigma]
end

function do_inference(model, df, iters)
    observations = make_observation(df.Y)
    (trace, _) = generate(model, (df,), observations)
    @showprogress 1 "Sampling..." for i = 1:iters
        line_params = select(model_symbols()...)
        (trace, _) = mh(trace, line_params)
    end
    return trace
end;


# cross validation
scores = cross_validate(
    (inds) -> do_inference(line_model, df[inds,:], 100000),
    (c, inds) -> rmse(
            df[inds,:Y],
            predict(line_model, (df[inds,:],), length(inds), c, model_symbols())
            ),
    length(df.Y),
    StratifiedKfold(df.KID, 2))

# get the mean and std of the scores
@show scores
@show mean_and_std(scores);

trace = do_inference(line_model, df, 100000);
@show trace[:a0]
@show trace[:sigma_a]
@show trace[(:ak, 1)]
@show trace[(:ak, 2)]
@show trace[(:ak, 3)]
@show trace[(:ak, 4)]
@show trace[:b0]
@show trace[:sigma_b]
@show trace[(:bk, 1)]
@show trace[(:bk, 2)]
@show trace[(:bk, 3)]
@show trace[(:bk, 4)]
@show trace[:sigma]


df.Y_pred = predict(line_model, (df,), length(df.Y), trace, model_symbols());
plt = lineplot(df.Y, df.Y);
for i in 1:4
    dff = @where(df, :KID .== i)
    scatterplot!(plt, dff.Y, dff.Y_pred);
end
println(plt)
