@time begin
    using CSV
    using Gen
    using UnicodePlots
    using ProgressMeter
end;

include("utils.jl")

df = CSV.read("/host/data/data-rental.txt")

@gen function log_model(areas::Vector{Float64})
    b1 = @trace(uniform(-10, 10), :b1)
    b2 = @trace(uniform(-10, 10), :b2)
    sigma = @trace(uniform(0, 1), :sigma)

    for (i, area) in enumerate(areas)
        mu = b1 + b2 * log10(area)
        @trace(normal(mu, sigma), (:y, i))
    end
end;


function do_inference(model, areas, ys, amount_of_computation)
    observations = make_observation(ys)
    (trace, _) = generate(model, (areas, ), observations)
    @showprogress 1 "Sampling..." for i = 1:amount_of_computation
        line_params = select(:b1, :b2, :sigma)
        (trace, _) = metropolis_hastings(trace, line_params)
    end
    return trace
end;

trace = do_inference(log_model, df.Area, log10.(df.Y), 200000);
@show trace[:b1]
@show trace[:b2]
@show trace[:sigma]

df.Y_pred = 10 .^ (trace[:b1] .+ log10.(df.Area) * trace[:b2]);
plt = lineplot(df.Y, df.Y);
println(scatterplot!(plt, df.Y, df.Y_pred));

plt = scatterplot(df.Area, df.Y_pred, xlim=[0, 100]);
println(scatterplot!(plt, df.Area, df.Y))

