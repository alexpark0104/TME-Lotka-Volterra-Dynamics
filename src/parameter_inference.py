import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
def tumor_immune_3sp(t, y, params):
    C, E, R = y
    p = params
    dC = (p['r'] * C * (1 - C / p['K'])
          + p['alpha'] * C * R
          - (p['beta'] * C * E) / (1 + p['eta'] * C))
    dE = (p['s']
          + (p['rho'] * C) / (1 + p['eps'] * R)
          - p['delta'] * C * E
          - p['gamma'] * E)
    dR = (p['xi']
          + p['sigma'] * C
          - p['phi'] * R)
    return [dC, dE, dR]
true_params = {
    'r': 0.15, 'K': 1000.0, 'eta': 0.005, 'alpha': 0.001,
    's': 0.1, 'eps': 0.05, 'delta': 0.001, 'gamma': 0.1,
    'xi': 0.1, 'sigma': 0.001, 'phi': 0.05,
    'beta': 0.03,   
    'rho': 0.05     
}
# Fixed parameters (assumed known / not fitted)
fixed = {k: true_params[k] for k in ['r', 'K', 'eta', 'alpha', 's', 'eps',
                                      'delta', 'gamma', 'xi', 'sigma', 'phi']}
# Initial conditions
y0_true = [50.0, 5.0, 5.0]
t_span = (0, 200)
t_data = np.linspace(0, 200, 21)  # 21 time points (e.g., every 10 days)

# Generate clean data
sol_true = solve_ivp(tumor_immune_3sp, t_span, y0_true,
                     args=(true_params,), t_eval=t_data, method='RK45')
C_clean = sol_true.y[0]
E_clean = sol_true.y[1]
R_clean = sol_true.y[2]

# Add noise (10% CV, Gaussian)
np.random.seed(42)
noise_level = 0.10
C_noisy = C_clean * (1 + noise_level * np.random.randn(len(t_data)))
E_noisy = E_clean * (1 + noise_level * np.random.randn(len(t_data)))
R_noisy = R_clean * (1 + noise_level * np.random.randn(len(t_data)))

# Ensure non-negative
C_noisy = np.maximum(C_noisy, 0)
E_noisy = np.maximum(E_noisy, 0)
R_noisy = np.maximum(R_noisy, 0)
def simulate_from_theta(theta, t_eval):
    """
    theta = [beta, rho, C0, E0, R0]
    """
    beta, rho, C0, E0, R0 = theta
    p = fixed.copy()
    p['beta'] = beta
    p['rho'] = rho

    y0 = [C0, E0, R0]
    sol = solve_ivp(tumor_immune_3sp, t_span, y0, args=(p,),
                    t_eval=t_eval, method='RK45', dense_output=True)

    if not sol.success:
        return np.full((3, len(t_eval)), 1e6)  # penalty for failed integration

    return sol.y  # shape (3, n_timepoints)


def residuals(theta):
    """
    Return flattened residuals: [C_resid, E_resid, R_resid]
    Weights: tumor gets 2x weight (usually measured more reliably)
    """
    sim = simulate_from_theta(theta, t_data)
    C_sim, E_sim, R_sim = sim[0], sim[1], sim[2]

    resid_C = (C_sim - C_noisy) / np.std(C_noisy)
    resid_E = (E_sim - E_noisy) / np.std(E_noisy)
    resid_R = (R_sim - R_noisy) / np.std(R_noisy)

    # Weight tumor 2x, effector 1x, Treg 0.5x (reflects measurement reliability)
    return np.concatenate([2.0 * resid_C, 1.0 * resid_E, 0.5 * resid_R])
theta0 = np.array([0.01, 0.01, 30.0, 2.0, 2.0])  # [beta, rho, C0, E0, R0]

# Bounds: keep parameters biologically reasonable
lb = np.array([0.001, 0.001, 10.0, 0.5, 0.5])
ub = np.array([0.20, 0.20, 200.0, 50.0, 50.0])

print("Starting parameter estimation...")
print(f"True values:  beta={true_params['beta']:.4f}, rho={true_params['rho']:.4f}, "
      f"C0={y0_true[0]:.1f}, E0={y0_true[1]:.1f}, R0={y0_true[2]:.1f}")
print(f"Initial guess: beta={theta0[0]:.4f}, rho={theta0[1]:.4f}, "
      f"C0={theta0[2]:.1f}, E0={theta0[3]:.1f}, R0={theta0[4]:.1f}")
result = least_squares(residuals, theta0, bounds=(lb, ub),
                       method='trf', max_nfev=500, verbose=2)

theta_fit = result.x
print(f"\nOptimization converged: {result.success}")
print(f"Cost: {result.cost:.4f}")
p_fit = fixed.copy()
p_fit['beta'] = theta_fit[0]
p_fit['rho'] = theta_fit[1]
y0_fit = theta_fit[2:5].tolist()

# High-resolution trajectories for plotting
t_fine = np.linspace(0, 200, 500)
sol_fit = solve_ivp(tumor_immune_3sp, t_span, y0_fit,
                    args=(p_fit,), t_eval=t_fine, method='RK45')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

labels = ['Cancer (C)', 'Effector T (E)', 'Treg (R)']
colors = ['red', 'blue', 'green']
clean_data = [C_clean, E_clean, R_clean]
noisy_data = [C_noisy, E_noisy, R_noisy]
fit_data = [sol_fit.y[i] for i in range(3)]

for i, ax in enumerate(axes):
    ax.plot(t_fine, fit_data[i], '-', color=colors[i], lw=2, label='Fitted')
    ax.plot(t_data, clean_data[i], 'o', color=colors[i], markersize=5,
            alpha=0.5, label='True (clean)')
    ax.plot(t_data, noisy_data[i], 'x', color='black', markersize=6,
            alpha=0.7, label='Noisy data')
    ax.set_title(labels[i])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Parameter Inference: True vs. Fitted', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
print("\n" + "="*60)
print("PARAMETER RECOVERY RESULTS")
print("="*60)
param_names = ['beta', 'rho', 'C0', 'E0', 'R0']
true_vals = [true_params['beta'], true_params['rho']] + y0_true
for name, true_val, fit_val in zip(param_names, true_vals, theta_fit):
    err = abs(fit_val - true_val) / true_val * 100
    print(f"{name:>8s}: True = {true_val:8.4f} | Fit = {fit_val:8.4f} | Error = {err:6.2f}%")
print("="*60)
