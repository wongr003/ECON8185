# This function discretizes an AR(1) process following Tauchen (1986) method
# y_{t+1} = μ + ρ * y_{t} + ϵ
# ϵ ~ N(0, σ^2)
# Y is the number of y states

using Distributions
function Tauchen(ρ, σ, Y, μ = 0, m = 3)
    if Y > 1
        ybar = μ / (1 - ρ);
        ymax = ybar + m * (σ / (1 - ρ^2))^(1 / 2); # maximum y
        ymin = ybar - m * (σ / (1 - ρ^2))^(1 / 2); # minimum y

        Δ = (ymax - ymin) / (Y - 1); # distance between each y
        y = ymin:Δ:ymax; # vector of possible states of y

        d = Normal();
        pdfY = ones(Y, Y); # preallocate memory and create the transition matrix below

        for i in 1:Y
            pdfY[i, 1] = cdf(d, (y[1] + Δ / 2 - ρ * y[i]) / σ^0.5);
            pdfY[i, Y] = 1 - cdf(d, (y[Y] - Δ / 2 - ρ * y[i]) / σ^0.5);
            for j in 2:Y-1
                pdfY[i, j] = cdf(d, (y[j] + Δ / 2 - ρ * y[i]) / σ^0.5) - cdf(d, (y[j] - Δ / 2 - ρ * y[i]) / σ^0.5);
            end
        end
    else
        y = μ;
        pdfY = 1;
    end

    return y, pdfY

end




