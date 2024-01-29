import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def dSdx(x, S):
    (V, n_MET, n_Kf, n_Kn, Cm) = S

    if x >= 100 and x < 300:
        mu = np.sin(x * (f / np.pi) * (np.pi / 180)) * 16 + 16
    else:
        mu = 0

    n_MET_inf = 1 / (1 + np.exp(-(mu - x0) / s1) * (1 + np.exp(-(mu - x0) / s0)))
    dn_MET = (n_MET_inf - n_MET) / tau_MET
    g_MET = n_MET * G_MET
    I_MET = g_MET * (V - EP)

    n_Kf_inf = 1 / (1 + np.exp(-(V - Vh_Kf) / s_Kf))
    tau_Kf = 12.3 + 0.5 * np.exp(-V / 39.9)
    dn_Kf = (n_Kf_inf - n_Kf) / tau_Kf
    g_Kf = n_Kf * G_Kf
    I_Kf = g_Kf * (V - E_K)

    n_Kn_inf = 1 / (1 + np.exp(-(V - Vh_Kn) / s_Kn))
    tau_Kn = 7.5 * (1 / (1 + np.exp(-(V + 92) / 13.6))) * np.exp(-V / 28.2)
    dn_Kn = (n_Kn_inf - n_Kn) / tau_Kn
    g_Kn = n_Kn * G_Kn
    I_Kn = g_Kn * (V - E_K)

    Cm = Clin + Qmax / (
        c * np.exp((V - Vh_Pres) / c) * (1 + np.exp(-1 * (V - Vh_Pres) / c)) ** 2
    )

    dV = -(I_MET + I_Kf + I_Kn) / Cm

    return [dV, dn_MET, dn_Kf, dn_Kn, Cm]


### initial values
# Ref: The effects of the activation of the inner-hair-cell basolateral K+ channels on auditory nerve responses (2018) Hearing Res
EP = 90
G_MET = 30
x0 = 20
s0 = 16
s1 = 35
tau_MET = 0.050

# Ref: Differential expression of outer hair cell potassium currents in the isolated cochlea of the guinea-pig (1996) J Physiol
G_Kf = 1
G_Kn = 50
Vh_Kf = -24
Vh_Kn = -92
s_Kf = 6.4
s_Kn = 17
E_K = -92

# set initial variable
Clin = 4
V0 = -84
f = 1000

# set time variable
dt = 0.1
t_end = 400
x = np.arange(0, t_end, dt)

mu_input = []
for i in range(len(x)):
    if x[i] >= 100 and x[i] < 300:
        mu_input.append(np.sin(x[i] * (f / np.pi) * (np.pi / 180)) * 16 + 16)
    else:
        mu_input.append(0)


###### WT #####
c = 25.76
Qmax = 366.2 * Clin
Vh_Pres = -37.02

n_MET_inf = 1 / (1 + np.exp(-(20 - x0) / s1) * (1 + np.exp(-(20 - x0) / s0)))
n_Kf_inf = 1 / (1 + np.exp(-(V0 - Vh_Kf) / s_Kf))
n_Kn_inf = 1 / (1 + np.exp(-(V0 - Vh_Kn) / s_Kn))
Cm_0 = Clin + Qmax / (
    c * np.exp((V0 - Vh_Pres) / c) * (1 + np.exp(-1 * (V0 - Vh_Pres) / c)) ** 2
)

S_0 = [V0, n_MET_inf, n_Kf_inf, n_Kn_inf, Cm_0]
mu_integ = []
solveODE = odeint(dSdx, t=x, y0=S_0, tfirst=True)
V_sol_WT = solveODE.T[0]
n_MET_sol_WT = solveODE.T[1]
n_Kf_sol_WT = solveODE.T[2]
n_Kn_sol_WT = solveODE.T[3]
Cm_sol_WT = solveODE.T[4]

Cm_sol_WT_diff = []
for i in range(len(x) - 1):
    Cm_sol_WT_diff.append((Cm_sol_WT[i + 1] - Cm_sol_WT[i]) / dt)

power_WT = []
work = 0
work_WT = []
for i in range(len(x) - 1):
    power_WT.append(0.5 * (Cm_sol_WT_diff[i] - Cm_0) * (V_sol_WT[i] - V0) ** 2 / 1000)
    work += abs(0.5 * (Cm_sol_WT_diff[i] - Cm_0) * (V_sol_WT[i] - V0) ** 2 / 1000 * dt)
    work_WT.append(work / 1000)

"""
power_freq_WT = []
for i in range(100, 10001, 100):
    f = i
    solveODE = odeint(dSdx, t=x, y0=S_0, tfirst=True)
    Cm_tmp = solveODE.T[4]
    Cm_tmp_diff = []
    for i in range(len(x) - 1):
        Cm_tmp_diff.append((Cm_tmp[i + 1] - Cm_tmp[i]) / dt)
    Cm_tmp_diff_max = max(Cm_tmp_diff[2500:3000])
    V_tmp_max = max(solveODE.T[0][2500:3000])
    power_freq_WT.append(0.5 * (Cm_tmp_diff_max - Cm_0) * (V_tmp_max - V0) ** 2)
    print(f)
"""

###### KO #####
c = 29.10
Qmax = 119.5 * Clin
Vh_Pres = -47.51

n_MET_inf = 1 / (1 + np.exp(-(20 - x0) / s1) * (1 + np.exp(-(20 - x0) / s0)))
n_Kf_inf = 1 / (1 + np.exp(-(V0 - Vh_Kf) / s_Kf))
n_Kn_inf = 1 / (1 + np.exp(-(V0 - Vh_Kn) / s_Kn))
Cm_0 = Clin + Qmax / (
    c * np.exp((V0 - Vh_Pres) / c) * (1 + np.exp(-1 * (V0 - Vh_Pres) / c)) ** 2
)

S_0 = [V0, n_MET_inf, n_Kf_inf, n_Kn_inf, Cm_0]
mu_integ = []
solveODE = odeint(dSdx, t=x, y0=S_0, tfirst=True)
V_sol_KO = solveODE.T[0]
n_MET_sol_KO = solveODE.T[1]
n_Kf_sol_KO = solveODE.T[2]
n_Kn_sol_KO = solveODE.T[3]
Cm_sol_KO = solveODE.T[4]

Cm_sol_KO_diff = []
for i in range(len(x) - 1):
    Cm_sol_KO_diff.append((Cm_sol_KO[i + 1] - Cm_sol_KO[i]) / dt)

power_KO = []
work = 0
work_KO = []
for i in range(len(x) - 1):
    power_KO.append(0.5 * (Cm_sol_KO_diff[i] - Cm_0) * (V_sol_KO[i] - V0) ** 2 / 1000)
    work += abs(0.5 * (Cm_sol_KO_diff[i] - Cm_0) * (V_sol_KO[i] - V0) ** 2 / 1000 * dt)
    work_KO.append(work / 1000)

"""
power_freq_KO = []
for i in range(100, 10001, 100):
    f = i
    solveODE = odeint(dSdx, t=x, y0=S_0, tfirst=True)
    Cm_tmp = solveODE.T[4]
    Cm_tmp_diff = []
    for i in range(len(x) - 1):
        Cm_tmp_diff.append((Cm_tmp[i + 1] - Cm_tmp[i]) / dt)
    Cm_tmp_diff_max = max(Cm_tmp_diff[2500:3000])
    V_tmp_max = max(solveODE.T[0][2500:3000])
    power_freq_KO.append(0.5 * (Cm_tmp_diff_max - Cm_0) * (V_tmp_max - V0) ** 2)
    print(f)
"""

# plot result
plt.subplot(331)
plt.plot(x, mu_input, "black", linewidth=1, label="f = 1000 Hz")
plt.legend(fontsize=8)
plt.xlabel("Time (ms)")
plt.ylabel("Stimulation (nm)")

plt.subplot(332)
plt.plot(x, n_MET_sol_WT, "black", linewidth=1)
plt.plot(x, n_MET_sol_KO, "orangered", linewidth=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("$n_{MET}$")
plt.ylim([0, 1])

plt.subplot(333)
plt.plot(x, V_sol_WT, "black", linewidth=1, label="WT")
plt.plot(x, V_sol_KO, "orangered", linewidth=0.5, label="KO")
plt.legend(fontsize=8)
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")

plt.subplot(334)
plt.plot(x, n_Kn_sol_WT, "black", linewidth=1, label="$K_{n}$")
plt.plot(x, n_Kn_sol_KO, "orangered", linewidth=0.5)
plt.plot(x, n_Kf_sol_WT, "black", linewidth=1, linestyle="--", label="$K_{f}$")
plt.plot(x, n_Kf_sol_KO, "orangered", linewidth=0.5)
plt.legend(fontsize=8)
plt.xlabel("Time (ms)")
plt.ylabel("$n_{K}$")
plt.ylim([0, 1])

plt.subplot(335)
plt.plot(x[:-1], Cm_sol_WT_diff, "black", linewidth=1)
plt.plot(x[:-1], Cm_sol_KO_diff, "orangered", linewidth=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("$C_{m}$ (pF)")

plt.subplot(336)
plt.plot(x[:-1], power_WT, "black", linewidth=1)
plt.plot(x[:-1], power_KO, "orangered", linewidth=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power (fJ)")

"""
plt.subplot(337)
plt.plot(range(100, 10001, 100), power_freq_WT, "black", linewidth=1)
plt.plot(range(100, 10001, 100), power_freq_KO, "orangered", linewidth=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xscale("log")

plt.subplot(338)
power_freq_div = []
for i in range(len(power_freq_WT)):
    power_freq_div.append(power_freq_KO[i] / power_freq_WT[i])
plt.plot(range(100, 10001, 100), power_freq_div, "black", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("$P_{KO}$/$P_{WT}$")
plt.ylim([0, 1])
plt.xscale("log")
"""

print(work_KO[-1] / work_WT[-1])
plt.show()


f = open("__output_WT__.txt", "w")
for i in range(len(x) - 1):
    f.write(
        "%.1f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n"
        % (
            x[i],
            mu_input[i],
            V_sol_WT[i],
            n_MET_sol_WT[i],
            n_Kf_sol_WT[i],
            n_Kn_sol_WT[i],
            Cm_sol_WT_diff[i],
            power_WT[i],
        )
    )
f.close()


f = open("__output_KO__.txt", "w")
for i in range(len(x) - 1):
    f.write(
        "%.1f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n"
        % (
            x[i],
            mu_input[i],
            V_sol_KO[i],
            n_MET_sol_KO[i],
            n_Kf_sol_KO[i],
            n_Kn_sol_KO[i],
            Cm_sol_KO_diff[i],
            power_KO[i],
        )
    )
f.close()

"""
f = open("__output_power__.txt", "w")
for i in range(len(range(100, 10001, 100))):
    f.write(
        "%d\t%.5f\t%.5f\t%.5f\n"
        % (
            range(100, 10001, 100)[i],
            power_freq_WT[i],
            power_freq_KO[i],
            power_freq_div[i],
        )
    )
f.close()
"""
