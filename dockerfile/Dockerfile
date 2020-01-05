FROM julia:latest

RUN julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/probcomp/Gen"));using Gen'
RUN julia -e 'using Pkg; Pkg.add("CSV");using CSV'
RUN julia -e 'using Pkg; Pkg.add("MLDataUtils");using MLDataUtils'
RUN julia -e 'using Pkg; Pkg.add("DataFramesMeta");using DataFramesMeta'
RUN julia -e 'using Pkg; Pkg.add("UnicodePlots");using UnicodePlots'
RUN julia -e 'using Pkg; Pkg.add("DataFrames");using DataFrames'
RUN julia -e 'using Pkg; Pkg.add("HTTP");using HTTP'
RUN julia -e 'using Pkg; Pkg.add("Distributions");using Distributions'
RUN julia -e 'using Pkg; Pkg.add("ProgressMeter");using ProgressMeter'
RUN julia -e 'using Pkg; Pkg.add("MLBase");using MLBase'
