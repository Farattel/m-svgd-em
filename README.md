<h1 align="center">Momentum SVGD-EM for Accelerated Maximum Marginal Likelihood Estimation</h1>
<h3 align="center">AISTATS 2026 (poster)</h3>
<p align="center">
  <a href="https://YOUR-LINK-HERE">Adam Rozzio</a> ·
  <a href="https://statml.io/students/rafael-athanasiades/">Rafael Athanasiades</a> ·
  <a href="https://odakyildiz.com/">O. Deniz Akyildiz</a>
</p>

---

## Description

This repository contains code for **M-SVGD-EM (Momentum Stein Variational Gradient Descent Expectation–Maximisation)**, an accelerated particle-based EM algorithm for maximum likelihood training of latent variable models.

---

## Example applications
We demonstrate M-SVGD-EM on benchmark problems:
1. Toy Hierarchical Models  
2. Bayesian Neural Networks (MNIST dataset)  
3. Bayesian Logistic Regression (Wisconsin Breast Cancer dataset)

---

## Running the code (local)
This repository is intended to be run locally on your machine after installing dependencies and installing the package in editable mode.

---
This project is built on top of the [**CoinEM**](https://github.com/chris-nemeth/CoinEM) project. The original codebase provides several particle-based EM algorithms and optimizers, including:
- **CoinEM**
- **SVGD-EM** (Stein Variational Gradient Descent EM)
- **SOUL** (Stochastic Optimisation via Unadjusted Langevin)
- **PGD** (Particle Gradient Descent)

## Installation
**Requirements:** Python 3.8–3.11

```bash
# Install all dependencies
pip install -r requirements.txt

# Install the CoinEM package (editable)
pip install -e .
