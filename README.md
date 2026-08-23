# Phenomenological Tumor-Immune-Treg Ecosystem Dynamics

## Overview
This repository contains a 3-species ordinary differential equation (ODE) framework modeling the tumor microenvironment (TME). It upgrades classical Lotka-Volterra predator-prey dynamics by introducing resource limitations and immunosuppressive checkpoints to model the competition between Cancer Cells (C), CD8+ Effector T-cells (E), and Regulatory T-cells (R).

## Mathematical Architecture
To correct the infinite growth assumption of standard models, the cancer cell population is constrained by a Logistic Carrying Capacity ($K$):
* **Cancer Kinetics:** 
  $$\frac{dC}{dt} = r C \left(1 - \frac{C}{K}\right) + \alpha C R - \frac{\beta C E}{1 + \eta C}$$

* **CD8+ Effector T-cells:** 
  $$\frac{dE}{dt} = s + \frac{\rho C}{1 + \epsilon R} - \delta C E - \gamma E - \mu C E$$
  *(Note: $\mu C E$ represents active immune exclusion by the tumor).*

* **Regulatory T-cells:**
  $$\frac{dR}{dt} = \xi + \sigma C - \phi R$$

## Validation
The model parameters were qualitatively validated against published *in vivo* AstraZeneca CT-26 mouse data (PMC6755954), successfully reproducing the directional shifts in CD8+ and Treg populations following combination anti-PD-L1 and anti-CTLA-4 checkpoint blockade.