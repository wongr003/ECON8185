function KalmanFilter(T::Matrix, Z::Matrix, Q::Matrix, H::Matrix, n::Integer, α0, y)
    
    # Initialize state variables
    αhat = repeat([zeros(size(α0))], n);
    αhat[1] = T*α0;

    println("I'm here 1")

    yhat = repeat([zeros(size(y[1]))], n);
    v = copy(yhat);

    P0 = [2.0]; # Initial guess given in the problem
    
    P = repeat([zeros(size(Q))], n);
    P[1] = T * P0 * T' + Q;

    println("I'm here 2, P[1] = $(P[1])")

    F = repeat([zeros(size(H))], n)
    G = repeat([zeros(size(Z * Q))], n)
    K = repeat([zeros(size(T * G[1]' * ones(size(H))))], n)

    for i in 1:n-1
        yhat[i] = Z * αhat[i];
        v[i] = y[i] - yhat[i];

        F[i] = Z * P[i] * Z' + H;
        G[i] = Z * P[i];
        K[i] = T * G[i]' * inv(F[i]);

        αhat[i + 1] = T * αhat[i] + K[i] * v[i];
        P[i + 1] = T * (P[i] - G[i]' * inv(F[i]) * G[i]) * T' + Q;
    end

    yhat[n] = Z * αhat[n];
    v[n] = y[n] - yhat[n];
    
    F[n] = Z * P[n] * Z' + H;
    G[n] = Z * P[n];
    K[n] = T * G[n]' * inv(F[n]);

    return αhat, v, yhat, P, F, G, K
end