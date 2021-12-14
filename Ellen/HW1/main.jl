## Packages
using Plots

## Files
include("LQ_Vaughan.jl")

md = Model();
kss,hss,lss,css = getSS(md);
A_tld,B_tld,Q_tld,R,W = getM(md,kss,hss);
P_LQ,F_LQ = LQ(A_tld,B_tld,R,Q_tld);
Pv,Fv = Vaughan(A_tld,B_tld,Q_tld,R,W);

## Plotting
# Constructing Policy function
nk = 100;
kGrid = LinRange(0.5*kss, 1.5*kss, nk);

# Low shock
pol_L_LQ = zeros(2, 1, 100);
pol_L_v = zeros(2, 1, 100);
for i = 1:nk
    pol_L_LQ[:,:,i] = -F_LQ*[kGrid[i]; -0.5; 1];
    pol_L_v[:,:,i] = -Fv*[kGrid[i]; -0.5; 1];
end

kpol_L_LQ = zeros(1,nk);
hpol_L_LQ = zeros(1,nk);
kpol_L_v = zeros(1,nk);
hpol_L_v = zeros(1,nk);

for i = 1:nk
    kpol_L_LQ[1,i] = pol_L_LQ[1,:,i][1];
    hpol_L_LQ[1,i] = pol_L_LQ[2,:,i][1];
    kpol_L_v[1,i] = pol_L_v[1,:,i][1];
    hpol_L_v[1,i] = pol_L_v[2,:,i][1];
end

# High shock
pol_H_LQ = zeros(2, 1, 100);
pol_H_v = zeros(2, 1, 100);

for i = 1:nk
    pol_H_LQ[:,:,i] = -F_LQ*[kGrid[i]; 0.5; 1];
    pol_H_v[:,:,i] = -Fv*[kGrid[i]; 0.5; 1];
end

kpol_H_LQ = zeros(1,nk);
hpol_H_LQ = zeros(1,nk);
kpol_H_v = zeros(1,nk);
hpol_H_v = zeros(1,nk);

for i = 1:nk
    kpol_H_LQ[1,i] = pol_H_LQ[1,:,i][1];
    hpol_H_LQ[1,i] = pol_H_LQ[2,:,i][1];
    kpol_H_v[1,i] = pol_H_v[1,:,i][1];
    hpol_H_v[1,i] = pol_H_v[2,:,i][1];
end

# SS shock
pol_SS_LQ = zeros(2, 1, 100);
pol_SS_v = zeros(2, 1, 100);

for i = 1:nk
    pol_SS_LQ[:,:,i] = -F_LQ*[kGrid[i]; 0.0; 1];
    pol_SS_v[:,:,i] = -Fv*[kGrid[i]; 0.0; 1];
end

kpol_SS_LQ = zeros(1,nk);
hpol_SS_LQ = zeros(1,nk);
kpol_SS_v = zeros(1,nk);
hpol_SS_v = zeros(1,nk);

for i = 1:nk
    kpol_SS_LQ[1,i] = pol_SS_LQ[1,:,i][1];
    hpol_SS_LQ[1,i] = pol_SS_LQ[2,:,i][1];
    kpol_SS_v[1,i] = pol_SS_v[1,:,i][1];
    hpol_SS_v[1,i] = pol_SS_v[2,:,i][1];
end

################### Plots ##################################
plot(kGrid,kGrid,label = "", color = :black, linestyle = :dash)
plot!(kGrid,vec(kpol_L_LQ),label = "Low",legend=:topleft)
plot!(kGrid,vec(kpol_H_LQ),label = "High")
plot!(kGrid,vec(kpol_SS_LQ),label = "SS")
plot!(kGrid,vec(kpol_L_v),label = "Low",legend=:topleft)
plot!(kGrid,vec(kpol_H_v),label = "High")
plot!(kGrid,vec(kpol_SS_v),label = "SS")

plot(kGrid,vec(hpol_L),label = "Low",legend=:topright)
plot!(kGrid,vec(hpol_H),label = "High")
plot!(kGrid,vec(hpol_SS),label = "SS")