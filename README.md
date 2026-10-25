# Phenomenological Tumor-Immune-Treg Ecosystem Dynamics

## Overview

This repository contains a 3-species ordinary differential equation (ODE) framework for exploring tumor-microenvironment (TME) dynamics. The model extends classical Lotka-Volterra predator-prey dynamics by introducing resource limitation, saturating immune-mediated tumor killing, and phenomenological immunosuppressive checkpoint effects.

The three modeled populations are Cancer Cells (`C`), CD8+ Effector T-cells (`E`), and Regulatory T-cells (`R`).

## Mathematical Architecture

To avoid the unbounded growth assumption in the classical Lotka-Volterra model, the cancer-cell population is constrained by a logistic carrying capacity (`K`):

- **Cancer Kinetics:**
  $$
  \frac{dC}{dt} = rC\left(1 - \frac{C}{K}\right) + \alpha CR - \frac{\beta CE}{1 + \eta C}
  $$

- **CD8+ Effector T-cell Dynamics:**
  $$
  \frac{dE}{dt} = s + \frac{\rho C}{1 + \epsilon R} - \delta CE - \gamma E
  $$

- **Regulatory T-cell Dynamics:**
  $$
  \frac{dR}{dt} = \xi + \sigma C - \phi R
  $$

Here, `C` represents cancer cells, `E` represents CD8+ effector T-cells, and `R` represents regulatory T-cells. The term `K` controls the tumor carrying capacity, while the saturating term in the cancer equation limits the effect of immune-mediated killing at higher tumor burden.

The current implementation does not include a separate `-\mu CE` immune-exclusion term. If active immune exclusion is added in a future version, the equation, parameter dictionary, and documentation should be updated together.

## Therapy Simulations

The therapy module introduces simplified anti-PD-1 and anti-CTLA-4 effects using periodic dosing, exponential drug decay, and saturating Emax-like responses. In the current phenomenological formulation:

- Anti-PD-1 reduces tumor-associated effector-cell loss.
- Anti-CTLA-4 reduces Treg-mediated suppression and increases effector-cell recruitment.

These therapy mechanisms are intended to explore qualitative model behavior. They are not calibrated pharmacokinetic or pharmacodynamic predictions.

## Parameter Inference

The parameter-inference example generates synthetic trajectories from the model, adds noise, and estimates selected parameters using nonlinear least squares. This demonstrates parameter-recovery behavior and should not be interpreted as a fit to independent biological measurements.

## Validation and Limitations

The repository includes an exploratory comparison with a small set of published immune-cell measurements. The comparison is qualitative because the model outputs are rescaled to an untreated reference point and the available observations are sparse.

A quantitative biological validation would require a verified primary CT-26 checkpoint-blockade source, documented data extraction, additional time points, tumor-burden measurements, uncertainty estimates, and a clearly defined fitting and validation protocol. The PMCID previously listed in this README should be verified before being used as support for the AstraZeneca CT-26 claim.

The model is a phenomenological research prototype and does not represent all relevant tumor, immune, stromal, or pharmacological mechanisms.
