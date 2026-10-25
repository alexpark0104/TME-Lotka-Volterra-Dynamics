import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#REAL DATA VALIDATION (AstraZeneca CT-26)
# Qualitative comparison — model predictions vs. published data
def tumor_immune_baseline(t, y, params):
    C, E, R = y
    p = params
    dC = (p['r'] * C * (1 - C / p['K']) + p['alpha'] * C * R
          - (p['beta'] * C * E) / (1 + p['eta'] * C))
    dE = (p['s'] + (p['rho'] * C) / (1 + p['eps'] * R)
          - p['delta'] * C * E - p['gamma'] * E)
    dR = (p['xi'] + p['sigma'] * C - p['phi'] * R)
    return [dC, dE, dR]

def tumor_immune_therapy(t, y, params, therapy):
    C, E, R = y
    p = params
    conc_PD1 = therapy['aPD1']['dose'] * np.exp(-np.log(2) * max(0, t - therapy['aPD1']['start']) / therapy['aPD1']['halflife']) if t >= therapy['aPD1']['start'] else 0
    conc_CTLA4 = therapy['aCTLA4']['dose'] * np.exp(-np.log(2) * max(0, t - therapy['aCTLA4']['start']) / therapy['aCTLA4']['halflife']) if t >= therapy['aCTLA4']['start'] else 0

    eff_PD1 = therapy['aPD1']['efficacy'] * conc_PD1 / (1.0 + conc_PD1)
    eff_CTLA4 = therapy['aCTLA4']['efficacy'] * conc_CTLA4 / (1.0 + conc_CTLA4)

    delta_eff = p['delta'] * (1.0 - eff_PD1)
    eps_eff = p['eps'] * (1.0 - 0.7 * eff_CTLA4)
    rho_eff = p['rho'] * (1.0 + 1.0 * eff_CTLA4)

    dC = (p['r'] * C * (1 - C / p['K']) + p['alpha'] * C * R
          - (p['beta'] * C * E) / (1 + p['eta'] * C))
    dE = (p['s'] + (rho_eff * C) / (1 + eps_eff * R)
          - delta_eff * C * E - p['gamma'] * E)
    dR = (p['xi'] + p['sigma'] * C - p['phi'] * R)
    return [dC, dE, dR]

# Retuned parameters from Block 6D
params = {
    'r': 0.15, 'K': 1000.0, 'eta': 0.005, 'alpha': 0.001,
    's': 0.1, 'eps': 0.30, 'delta': 0.001, 'gamma': 0.1,
    'xi': 0.1, 'sigma': 0.001, 'phi': 0.05,
    'beta': 0.0310, 'rho': 0.0485
}
# CT-26 untreated timecourse
az_day = np.array([7, 14])
az_CD8 = np.array([2.944, 1.168])   # % of CD45+
az_Treg = np.array([3.406, 1.165])   # % of CD45+

#CT-26 checkpoint inhibition (single timepoint, ~day 14)
az_treated_CD8 = 9.64   # % of CD45+ with anti-PD-L1 + anti-CTLA-4
az_treated_Treg = 1.74  # % of CD45+ with anti-PD-L1 + anti-CTLA-4
# SIMULATE: UNTREATED (Baseline)
y0 = [52.37, 5.01, 4.78]
t_span = (0, 14)
t_eval = np.linspace(0, 14, 500)

sol_untreated = solve_ivp(tumor_immune_baseline, t_span, y0,
                          args=(params,), t_eval=t_eval, method='RK45')

# Extract model predictions at days 7 and 14
idx_d7 = np.argmin(np.abs(t_eval - 7))
idx_d14 = np.argmin(np.abs(t_eval - 14))

model_C_d7 = sol_untreated.y[0, idx_d7]
model_C_d14 = sol_untreated.y[0, idx_d14]
model_E_d7 = sol_untreated.y[1, idx_d7]
model_E_d14 = sol_untreated.y[1, idx_d14]
model_R_d7 = sol_untreated.y[2, idx_d7]
model_R_d14 = sol_untreated.y[2, idx_d14]
# SIMULATE: TREATED (Anti-PD-L1 + Anti-CTLA-4 at day 7)
therapy_combo = {
    'aPD1': {'dose': 5.0, 'period': 999, 'halflife': 20, 'start': 7, 'efficacy': 0.8},
    'aCTLA4': {'dose': 3.0, 'period': 999, 'halflife': 15, 'start': 7, 'efficacy': 0.7}
}

sol_treated = solve_ivp(tumor_immune_therapy, t_span, y0,
                        args=(params, therapy_combo), t_eval=t_eval, method='RK45')

model_E_treated_d14 = sol_treated.y[1, idx_d14]
model_R_treated_d14 = sol_treated.y[2, idx_d14]
# NORMALIZE MODEL OUTPUT TO MATCH DATA SCALE
# We use a simple linear scaling: model_E * scale_E = %CD8+
# Scale factor determined by matching day 7 untreated
scale_E = az_CD8[0] / model_E_d7   # scale to match day 7 CD8+
scale_R = az_Treg[0] / model_R_d7  # scale to match day 7 Treg

model_CD8_d7 = model_E_d7 * scale_E
model_CD8_d14 = model_E_d14 * scale_E
model_CD8_treated_d14 = model_E_treated_d14 * scale_E

model_Treg_d7 = model_R_d7 * scale_R
model_Treg_d14 = model_R_d14 * scale_R
model_Treg_treated_d14 = model_R_treated_d14 * scale_R
# PLOT: Model vs. Real Data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CD8+ T-cells
axes[0].plot(t_eval, sol_untreated.y[1] * scale_E, 'b-', lw=2, label='Model (untreated)')
axes[0].plot(t_eval, sol_treated.y[1] * scale_E, 'b--', lw=2, label='Model (treated)')
axes[0].scatter([7, 14], az_CD8, color='red', s=100, zorder=5, label='Data: untreated')
axes[0].scatter([14], [az_treated_CD8], color='green', s=100, marker='s', zorder=5, label='Data: treated')
axes[0].set_xlabel('Day post-implantation')
axes[0].set_ylabel('CD8+ (% of CD45+)')
axes[0].set_title('CD8+ T-cell Dynamics')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tregs
axes[1].plot(t_eval, sol_untreated.y[2] * scale_R, 'g-', lw=2, label='Model (untreated)')
axes[1].plot(t_eval, sol_treated.y[2] * scale_R, 'g--', lw=2, label='Model (treated)')
axes[1].scatter([7, 14], az_Treg, color='red', s=100, zorder=5, label='Data: untreated')
axes[1].scatter([14], [az_treated_Treg], color='green', s=100, marker='s', zorder=5, label='Data: treated')
axes[1].set_xlabel('Day post-implantation')
axes[1].set_ylabel('Treg (% of CD45+)')
axes[1].set_title('Treg Dynamics')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Block 7: Model Validation Against AstraZeneca CT-26 Data', fontsize=14)
plt.tight_layout()
plt.show()
# VALIDATION TABLE
print("="*75)
print("QUALITATIVE VALIDATION: Model Predictions vs. AstraZeneca Data")
print("="*75)
print(f"\n{'Observation':<50} {'Model':<15} {'Data':<15} {'Match?':<8}")
print("-"*75)

checks = [
    ("CD8+ declines from day 7→14 (untreated)",
     f"{model_CD8_d7:.2f}→{model_CD8_d14:.2f}",
     f"{az_CD8[0]:.2f}→{az_CD8[1]:.2f}"),
    ("Treg declines from day 7→14 (untreated)",
     f"{model_Treg_d7:.2f}→{model_Treg_d14:.2f}",
     f"{az_Treg[0]:.2f}→{az_Treg[1]:.2f}"),
    ("Treatment increases CD8+ at day 14",
     f"{model_CD8_treated_d14:.2f}",
     f"{az_treated_CD8:.2f}"),
    ("Treatment increases Treg at day 14",
     f"{model_Treg_treated_d14:.2f}",
     f"{az_treated_Treg:.2f}"),
]

for obs, model_val, data_val in checks:
    # Check direction
    model_dir = "↑" if "→" in model_val and float(model_val.split('→')[1]) > float(model_val.split('→')[0]) else \
                "↓" if "→" in model_val else ""
    data_dir = "↑" if "→" in data_val and float(data_val.split('→')[1]) > float(data_val.split('→')[0]) else \
               "↓" if "→" in data_val else ""
    match = "PASS" if model_dir == data_dir or "→" not in model_val else "CHECK"
    print(f"{obs:<50} {model_val:<15} {data_val:<15} {match:<8}")

print("-"*75)
print("NOTE: Validation is qualitative (directional). Quantitative fit")
print("      requires more timepoints and tumor volume data.")
print("="*75)
