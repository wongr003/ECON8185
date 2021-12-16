## Packages
using Plots

## Files
include("VFI.jl")
include("LQ_Vaughan.jl")

################################### Running three methods ########################
md = Model();
kss,hss,lss,css = getSS(md);

## Construct capital grid
nk = 1000; # number of capital grid
kmin = 0.5*kss;
kmax = 1.5*kss;
kGrid = LinRange(kmin,kmax,nk);

V_VFI,kpol_VFI,hpol_VFI,cpol_VFI = VFI(md,nk,kGrid)

A_tld,B_tld,Q_tld,R,W = getM(md,kss,hss);
P_LQ,F_LQ = LQ(A_tld,B_tld,R,Q_tld);
Pv,Fv = Vaughan(A_tld,B_tld,Q_tld,R,W);

################################### Plotting ########################
## VFI
plot(kGrid,V_VFI[:,1])
plot!(kGrid,V_VFI[:,2])
 
plot(kGrid,kGrid,label = "", color = :black, linestyle = :dash)
plot!(kGrid,kpol_VFI[:,1],label = "low",legend = :topleft)
plot!(kGrid,kpol_VFI[:,3],label = "SS",legend = :topleft)
plot!(kGrid,kpol_VFI[:,5],label = "high",legend = :topleft)

plot(kGrid,hpol[:,1],label = "low",legend = :topleft)
plot!(kGrid,hpol[:,3],label = "high",legend = :topleft)

plot(kGrid,cpol[:,1],label = "low",legend = :topleft)
plot!(kGrid,cpol[:,3],label = "high",legend = :topleft)

## LQ and Vaughan
# Low shock
pol_L_LQ = zeros(2, 1, nk);
pol_L_v = zeros(2, 1, nk);
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
pol_H_LQ = zeros(2, 1, nk);
pol_H_v = zeros(2, 1, nk);

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
pol_SS_LQ = zeros(2, 1, nk);
pol_SS_v = zeros(2, 1, nk);

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
# capital policy LQ
plot(kGrid,kGrid,label = "45 degree line", color = :black, linestyle = :dash,
title = "Optimal capital policy function (LQ)",xlabel = "k_{t}", ylabel = "k_{t+1}")
plot!(kGrid,vec(kpol_L_LQ),label = "Low_LQ",legend=:topleft)
plot!(kGrid,vec(kpol_H_LQ),label = "High_LQ")
plot!(kGrid,vec(kpol_SS_LQ),label = "SS_LQ")
savefig("figs/kpol_LQ.png")

# labor policy LQ
plot(kGrid,vec(hpol_L_LQ),label = "Low",legend=:topright,
title = "Optimal labor policy function (LQ)",xlabel = "k_{t}", ylabel = "h_{t}")
plot!(kGrid,vec(hpol_H_LQ),label = "High")
plot!(kGrid,vec(hpol_SS_LQ),label = "SS")
savefig("figs/hpol_LQ.png")

# capital policy Vaughan
plot(kGrid,kGrid,label = "45 degree line", color = :black, linestyle = :dash,
title = "Optimal capital policy function (Vaughan)",xlabel = "k_{t}", ylabel = "k_{t+1}")
plot!(kGrid,vec(kpol_L_v),label = "Low_Vaughan",legend=:topleft)
plot!(kGrid,vec(kpol_H_v),label = "High_Vaughan")
plot!(kGrid,vec(kpol_SS_v),label = "SS_Vaughan")
savefig("figs/kpol_Vaughan.png")

# labor policy Vaughan
plot(kGrid,vec(hpol_L_v),label = "Low",legend=:topright,
title = "Optimal labor policy function (Vaughan)",xlabel = "k_{t}", ylabel = "h_{t}")
plot!(kGrid,vec(hpol_H_v),label = "High")
plot!(kGrid,vec(hpol_SS_v),label = "SS")
savefig("figs/hpol_Vaughan.png")



