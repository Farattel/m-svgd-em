<h1 align="center">Momentum SVGD-EM for Accelerated Maximum Marginal Likelihood Estimation</h1>
<h3 align="center">AISTATS 2025 (poster)</h3>
<p align="center">
  <a href="https://YOUR-LINK-HERE">Adam Rozzio</a> ·
  <a href="https://statml.io/students/rafael-athanasiades/">Rafael Athanasiades</a> ·
  <a href="https://odakyildiz.com/">O. Deniz Akyildiz</a>
</p>

---

## Description
This repository contains code for **M-SVGD-EM (Momentum-accelerated Stein Variational Gradient Descent Expectation–Maximisation)**, an extension of particle-based EM algorithm SVGD-EM for maximum likelihood training of latent variable models.

---

## Our contribution: M-SVGD-EM

Main additions:
1. **AcceleratedSteinExpectationStep**  
   Extension of the standard `SteinExpectationStep` adding momentum acceleration to the E-step particle updates.

2. **AcceleratedMaximisationStep**  
   Momentum-accelerated parameter updates in the M-step.

3. **m_svgd_em function**  
   End-to-end M-SVGD-EM implementation integrating both accelerated steps, with independent control of acceleration rates for E and M (for simplicity, notebooks may use the same value for both).

Our implementation allows flexible configuration of acceleration parameters to tune performance for specific problems.

---

## Example applications
We demonstrate M-SVGD-EM on benchmark problems already provided in the original CoinEM codebase:
1. Toy Hierarchical Models  
2. Bayesian Neural Networks (MNIST dataset)  
3. Bayesian Logistic Regression (Wisconsin Breast Cancer dataset)



---

## Running the code (local)
This repository is intended to be run locally on your machine after installing dependencies and installing the package in editable mode.

If you are already familiar with the CoinEM workflows, you can run the same experiments and select the **M-SVGD-EM** functions where relevant.

---
This project is built on top of the [**CoinEM**](https://github.com/chris-nemeth/CoinEM) project. The original codebase provides several particle-based EM algorithms and optimizers, including:
- **CoinEM** (SVGD with CoCaBO optimizer)
- **SVGD-EM** (Stein Variational Gradient Descent EM)
- **SOUL** (Stochastic Langevin Dynamics)
- **PGD** (Particle Gradient Descent)

## Installation
**Requirements:** Python 3.8–3.11

```bash
# Install all dependencies
pip install -r requirements.txt

# Install the CoinEM package (editable)
pip install -e .
