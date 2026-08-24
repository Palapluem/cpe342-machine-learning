# CPE 342 Machine Learning

This repository contains coursework, assignments, lecture materials, and reference textbooks for the **CPE 342: Machine Learning** course (Semester 1/2024), Department of Computer Engineering, Faculty of Engineering, King Mongkut's University of Technology Thonburi (KMUTT).

---

## 📂 Repository Structure

```text
cpe342-machine-learning/
├── assignment/
│   ├── Assignment 1_Training models/
│   │   ├── Assignment_1_OLS.ipynb             # Jupyter Notebook implementation
│   │   ├── CPE342_Assignment 1_Problem.pdf     # Problem specification
│   │   ├── CPE342_Assignment 1_main.tex       # LaTeX source report
│   │   ├── CPE342_Assignment 1_main.pdf       # Compiled report
│   │   └── CPE342_Assignment 1_Full Result.pdf # Complete merged submission
│   │
│   └── Assignment 2_Training models via iterative approach/
│       ├── Assignment_2_GD.ipynb              # Jupyter Notebook implementation
│       ├── CPE342_Assignment 2_Data.csv       # Training dataset (n = 100)
│       ├── CPE342_Assignment 2_Problem.pdf    # Problem specification
│       ├── CPE342_Assignment 2_main.tex      # LaTeX source report
│       ├── CPE342_Assignment 2_main.pdf      # Compiled report
│       └── CPE342_Assignment 2_Full Result.pdf# Complete merged submission
│   │
│   └── Assignment 3_Regression/
│       ├── Assignment_3_Regression.ipynb      # Jupyter Notebook implementation
│       ├── Lab&Assignment_regression.ipynb    # Coursework notebook
│       ├── Telco-Churn.csv                    # Dataset (N = 7,043)
│       ├── CPE342_Assignment 3_main.tex      # LaTeX source report
│       ├── CPE342_Assignment 3_main.pdf      # Compiled report
│       └── plot_1_*.pdf to plot_5_*.pdf       # High-resolution vector figures
│
├── lecture/                                   # Lecture slides and demo notebooks
│   ├── Lecture 1–12 PDFs
│   ├── ML_2_Training_Models.ipynb
│   └── ML_3_SVM.ipynb
│
└── textbook/                                  # Reference textbooks (ISLR / O'Reilly)
```

---

## 📝 Coursework & Assignments

### [Assignment 1: Training Models (OLS Regression)](./assignment/Assignment%201_Training%20models/)
* **Topic:** Linear Regression using Ordinary Least Squares (OLS) via Closed-Form Analytic Solutions.
* **Key Tasks:**
  * Proof and derivation of Normal Equations ($\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^T\mathbf{y}$).
  * Solving for regression coefficients ($\alpha, \beta$) via multiple analytical methods (Cramer's Rule, Matrix Inversion, $S_{xx}/S_{xy}$ formulation).
  * Comprehensive regression diagnostics (Residual vs Fitted, Normal Q-Q, Scale-Location, Actual vs Predicted).
* **Deliverables:**
  * [`Assignment_1_OLS.ipynb`](./assignment/Assignment%201_Training%20models/Assignment_1_OLS.ipynb)
  * [`CPE342_Assignment 1_Full Result.pdf`](./assignment/Assignment%201_Training%20models/CPE342_Assignment%201_Full%20Result.pdf)

---

### [Assignment 2: Training Models via Iterative Approach (Gradient Descent)](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/)
* **Topic:** Non-linear Model Fitting using Batch Gradient Descent (GD).
* **Model Equation:** $\hat{y} = C_0 + C_1 e^{C_2 x}$
* **Key Tasks:**
  * Formulation of Mean Squared Error (MSE) loss function: $\mathcal{L}(C_0, C_1, C_2) = \frac{1}{2n}\sum (y_i - \hat{y}_i)^2$.
  * Mathematical derivation of partial derivatives/gradients ($\frac{\partial\mathcal{L}}{\partial C_0}, \frac{\partial\mathcal{L}}{\partial C_1}, \frac{\partial\mathcal{L}}{\partial C_2}$) using the Chain Rule.
  * Definition of simultaneous parameter update rules with learning rate $\eta$.
  * Python implementation from scratch using NumPy with early stopping.
  * Training diagnostic visualizations (Loss learning curve, parameter trajectories, fitted non-linear curve, residual distribution).
  * Mathematical evaluation via Coefficient of Determination ($R^2$).
* **Deliverables:**
  * [`Assignment_2_GD.ipynb`](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/Assignment_2_GD.ipynb)
  * [`CPE342_Assignment 2_Full Result.pdf`](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/CPE342_Assignment%202_Full%20Result.pdf)

---

### [Assignment 3: Regression — Survival Analysis (Telco Customer Churn)](./assignment/Assignment%203_Regression/)
* **Topic:** Time-to-Event Modeling and Non-Parametric Survival Analysis on Customer Churn Data ($N = 7,043$).
* **Key Tasks:**
  * Formulation of duration $T = \text{tenure}$ (months) and binary event indicator $E = \mathbb{I}(\text{Churn} == \text{'Yes'})$ with Right-Censoring handling.
  * Mathematical definitions and derivation of Survival Function $S(t)$, Hazard Function $\lambda(t)$, and Cumulative Hazard $\Lambda(t) = -\ln S(t)$ based on Lecture 4.
  * Non-parametric Kaplan-Meier estimation $\hat{S}(t) = \prod (1 - d_i/n_i)$ with Greenwood's 95% confidence intervals.
  * Comparative survival segmentation across payment methods (`Electronic check`, `Mailed check`, `Bank transfer (auto)`, `Credit card (auto)`).
  * Cumulative Hazard analysis via Nelson-Aalen estimator.
  * Hypothesis testing via Multivariate Log-Rank Test ($\chi^2 = 865.24, p = 3.07 \times 10^{-187}$) and Pairwise Log-Rank tests.
  * Milestone retention probabilities calculation ($t = 12, 24, 36, 48, 60, 72$ months) and business churn mitigation strategies.
* **Deliverables:**
  * [`Assignment_3_Regression.ipynb`](./assignment/Assignment%203_Regression/Assignment_3_Regression.ipynb)
  * [`CPE342_Assignment 3_main.pdf`](./assignment/Assignment%203_Regression/CPE342_Assignment%203_main.pdf)

---

## 🛠️ Environment & Prerequisites

* **Python:** 3.10+
* **Core Libraries:** `numpy`, `pandas`, `matplotlib`, `scipy`, `lifelines`, `statsmodels`, `seaborn`, `jupyter`
* **Typography:** TH Sarabun New / Sarabun font support for Matplotlib charts
* **Report Compilation:** XeLaTeX / TeX Live / MiKTeX (Polyglossia + Sarabun font)

---

## 👤 Author
* **Wisit Suwannao**
* Student ID: `67070501042`
* Department of Computer Engineering, KMUTT