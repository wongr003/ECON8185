using Distributions

function mc_sample_path(P; init = 1, sample_size = 1000)
    @assert size(P)[1] == size(P)[2] # square required
    N = size(P)[1]; # should be square
    
    # create vector of discrete RVs for each row of stochastic matrix P
    dists = [Categorical(P[i, :]) for i in 1:N]

    # set up Simulation
    X = fill(0, sample_size); # allocate memory, or zeros(Int64, sample_size)
    X[1] = init # set the initial state

    for t in 2:sample_size
        dist = dists[X[t - 1]];
        X[t] = rand(dist);
    end

    return X
end

