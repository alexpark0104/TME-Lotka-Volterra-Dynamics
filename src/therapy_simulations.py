import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def drug_concentration(t, dose, period, halflife, start_day=0):
    """
    Periodic bolus dosing with exponential decay.
    Returns drug concentration at time t.
    """
    if t < start_day:
        return 0.0
    # Number of doses given before time t
    n_doses = int((t - start_day) // period) + 1
    conc = 0.0
    for n in range(n_doses):
        t_dose = start_day + n * period
        if t >= t_dose:
            conc += dose * np.exp(-np.log(2) * (t - t_dose) / halflife)
    return conc


def tumor_immune_therapy(t, y, params, therapy):
    """
    3-species model WITH checkpoint inhibitor therapy.

    therapy dict:
        'aPD1': {'dose': ..., 'period': ..., 'halflife': ..., 'start': ..., 'efficacy': ...}
        'aCTLA4': {'dose': ..., 'period': ..., 'halflife': ..., 'start': ..., 'efficacy': ...}
    """
    C, E, R = y
    p = params

    # --- Drug concentrations ---
    conc_PD1 = drug_concentration(t,
                                   therapy['aPD1']['dose'],
                                   therapy['aPD1']['period'],
                                   therapy['aPD1']['halflife'],
                                   therapy['aPD1']['start'])
    conc_CTLA4 = drug_concentration(t,
                                     therapy['aCTLA4']['dose'],
                                     therapy['aCTLA4']['period'],
                                     therapy['aCTLA4']['halflife'],
                                     therapy['aCTLA4']['start'])

    # --- Drug effects (Emax-like) ---
    # Anti-PD-1: reduces exhaustion delta
    eff_PD1 = therapy['aPD1']['efficacy'] * conc_PD1 / (1.0 + conc_PD1)
    delta_eff = p['delta'] * (1.0 - eff_PD1)

    # Anti-CTLA-4: reduces Treg suppression eps, boosts recruitment rho
    eff_CTLA4 = therapy['aCTLA4']['efficacy'] * conc_CTLA4 / (1.0 + conc_CTLA4)
    eps_eff = p['eps'] * (1.0 - 0.7 * eff_CTLA4)   # 70% max reduction in suppression
    rho_eff = p['rho'] * (1.0 + 1.0 * eff_CTLA4)    # 100% max boost in recruitment

    # --- Dynamics ---
    dC = (p['r'] * C * (1 - C / p['K'])
          + p['alpha'] * C * R
          - (p['beta'] * C * E) / (1 + p['eta'] * C))

    dE = (p['s']
          + (rho_eff * C) / (1 + eps_eff * R)
          - delta_eff * C * E
          - p['gamma'] * E)

    dR = (p['xi']
          + p['sigma'] * C
          - p['phi'] * R)

    return [dC, dE, dR]
params = {
    'r': 0.15, 'K': 1000.0, 'eta': 0.005, 'alpha': 0.001,
    's': 0.1, 'eps': 0.05, 'delta': 0.001, 'gamma': 0.1,
    'xi': 0.1, 'sigma': 0.001, 'phi': 0.05,
    'beta': 0.0310,   # fitted value from Block 3
    'rho': 0.0485     # fitted value from Block 3
}

y0 = [52.37, 5.01, 4.78]  # fitted initial conditions
t_span = (0, 400)
t_eval = np.linspace(0, 400, 2000)
# Scenario 0: No therapy (control)
therapy_none = {
    'aPD1': {'dose': 0, 'period': 999, 'halflife': 1, 'start': 999, 'efficacy': 0},
    'aCTLA4': {'dose': 0, 'period': 999, 'halflife': 1, 'start': 999, 'efficacy': 0}
}

# Scenario 1: Anti-PD-1 monotherapy
therapy_PD1 = {
    'aPD1': {'dose': 5.0, 'period': 14, 'halflife': 20, 'start': 50, 'efficacy': 0.8},
    'aCTLA4': {'dose': 0, 'period': 999, 'halflife': 1, 'start': 999, 'efficacy': 0}
}

# Scenario 2: Anti-CTLA-4 monotherapy
therapy_CTLA4 = {
    'aPD1': {'dose': 0, 'period': 999, 'halflife': 1, 'start': 999, 'efficacy': 0},
    'aCTLA4': {'dose': 3.0, 'period': 21, 'halflife': 15, 'start': 50, 'efficacy': 0.7}
}

# Scenario 3: Combination therapy
therapy_combo = {
    'aPD1': {'dose': 5.0, 'period': 14, 'halflife': 20, 'start': 50, 'efficacy': 0.8},
    'aCTLA4': {'dose': 3.0, 'period': 21, 'halflife': 15, 'start': 50, 'efficacy': 0.7}
}

scenarios = [
    ('No Therapy', therapy_none, 'gray'),
    ('Anti-PD-1 Mono', therapy_PD1, 'orange'),
    ('Anti-CTLA-4 Mono', therapy_CTLA4, 'purple'),
    ('Combination', therapy_combo, 'green')
]
ig, axes = plt.subplots(2, 2, figsize=(14, 10))

for (name, therapy, color) in scenarios:
    sol = solve_ivp(tumor_immune_therapy, t_span, y0,
                    args=(params, therapy), t_eval=t_eval, method='RK45')

    C = sol.y[0]
    E = sol.y[1]
    R = sol.y[2]

    # Plot Cancer
    axes[0, 0].plot(t_eval, C, color=color, lw=2, label=name)
    # Plot Effector T
    axes[0, 1].plot(t_eval, E, color=color, lw=2, label=name)
    # Plot Treg
    axes[1, 0].plot(t_eval, R, color=color, lw=2, label=name)
    # Tumor + Therapy timeline
    axes[1, 1].plot(t_eval, C, color=color, lw=2, label=name)

    # Mark therapy start
    if therapy['aPD1']['dose'] > 0:
        axes[1, 1].axvline(therapy['aPD1']['start'], color=color, ls='--', alpha=0.3)
    if therapy['aCTLA4']['dose'] > 0:
        axes[1, 1].axvline(therapy['aCTLA4']['start'], color=color, ls=':', alpha=0.3)

# Formatting
axes[0, 0].set_title('Cancer Burden (C)', fontsize=12)
axes[0, 0].set_ylabel('Population')
axes[0, 0].legend(loc='upper left', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Effector T-cells (E)', fontsize=12)
axes[0, 1].set_ylabel('Population')
axes[0, 1].legend(loc='upper left', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_title('Regulatory T-cells (R)', fontsize=12)
axes[1, 0].set_xlabel('Time (days)')
axes[1, 0].set_ylabel('Population')
axes[1, 0].legend(loc='upper left', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_title('Tumor Response to Therapy', fontsize=12)
axes[1, 1].set_xlabel('Time (days)')
axes[1, 1].set_ylabel('Cancer Population')
axes[1, 1].legend(loc='upper left', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(50, color='black', ls='--', alpha=0.5, label='Therapy start')

plt.suptitle('Checkpoint Inhibitor Therapy Scenarios', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
print("\n" + "="*70)
print("THERAPY OUTCOME SUMMARY (at t=400 days)")
print("="*70)
print(f"{'Scenario':<20} {'Final C':>10} {'Final E':>10} {'Final R':>10} {'Tumor Change':>12}")
print("-"*70)

for (name, therapy, _) in scenarios:
    sol = solve_ivp(tumor_immune_therapy, t_span, y0,
                    args=(params, therapy), t_eval=[400], method='RK45')
    C_end, E_end, R_end = sol.y[:, -1]
    change = (C_end - y0[0]) / y0[0] * 100
    print(f"{name:<20} {C_end:>10.1f} {E_end:>10.1f} {R_end:>10.1f} {change:>11.1f}%")
print("="*70)