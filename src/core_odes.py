import numpy as np
from scipy.integrate import solve_ivp

def tumor_immune_3sp(t, y, params):
    """
    C : cancer cells
    E : effector CD8+ T-cells
    R : regulatory T-cells (Tregs)
    """
    C, E, R = y
    p = params

    # Cancer cell dynamics
    dC = (p['r'] * C * (1 - C / p['K']) + p['alpha'] * C * R - (p['beta'] * C * E) / (1 + p['eta'] * C))

    # Effector T-cell dynamics
    dE = (p['s'] + (p['rho'] * C) / (1 + p['eps'] * R) - p['delta'] * C * E - p['gamma'] * E)

    # Regulatory T-cell dynamics
    dR = (p['xi'] + p['sigma'] * C - p['phi'] * R)

    return [dC, dE, dR]