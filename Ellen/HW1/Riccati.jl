using LinearAlgebra

## Iterate Riccati difference equation
function Riccati(A_tld,B_tld,R,Q_tld,tol)

    P = ones(size(A_tld)[2],size(A_tld)[1]);
    F = ones(size(R)[1],size(W)[1]);

    pDiff = 10;
    fDiff = 10;
    iter = 0;

    while pDiff >= tol.*opnorm(P,1) && fDiff > tol.*opnorm(F,1)
        A = R+B_tld'*P*B_tld;

        P1 = Q_tld+A_tld'*P*A_tld-A_tld'*P*B_tld*(A\B_tld')*P*A_tld;
        pDiff = opnorm(P1-P,1);

        F1 = A\B_tld'*P*A_tld;
        fDiff = opnorm(F1-F,1);

        iter = iter + 1;

        P = copy(P1)
        F = copy(F1)
    end

    return P,F

end
