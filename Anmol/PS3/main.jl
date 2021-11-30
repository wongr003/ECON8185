include("functions.jl")

r0 = 0.01;
AA0, cpol0 = AiyagariEGM(r0);
abar0 = zeros(length(cpol0));

polC_ss,polA_ss,λ_ss,Ar_ss,K_ss,r_ss = equilibriumEGM(AA0,cpol0,abar0);


