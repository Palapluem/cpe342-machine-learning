import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# 1. Title & Problem Overview
cells.append(nbf.v4.new_markdown_cell(
r"""# CPE 342 Machine Learning — Assignment 2: Training Models
**Author:** 67070501042 วิศิษฐ์ สุวรรณเนาว์

## 1. Problem Overview
In this assignment, we use an **iterative approach** (Gradient Descent) to train a model and fit it to a dataset of 100 observations ($n=100$). 

The goal is to fit a quadratic model without a linear $x$ term:
$$ \hat{y} = C_0 + C_1 x^2 $$
Where:
- $C_0$ is the y-intercept (Bias)
- $C_1$ is the coefficient of $x^2$ (Slope of $x^2$)"""
))

# 2. Font & Library Setup
cells.append(nbf.v4.new_markdown_cell(
r"""## 2. Library & Font Setup
Loading libraries and downloading the Sarabun font to support Thai rendering in Matplotlib if needed."""
))
cells.append(nbf.v4.new_code_cell(
r"""# ดาวน์โหลดฟอนต์ TH Sarabun New สำหรับใช้ใน Matplotlib (รันบน Colab/Jupyter ได้)
!wget -q https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import urllib.request
import os

# การตั้งค่าฟอนต์
font_path = 'thsarabunnew-webfont.ttf'
if not os.path.exists(font_path):
    urllib.request.urlretrieve('https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf', font_path)
fm.fontManager.addfont(font_path)
mpl.rc('font', family='TH Sarabun New', size=14)
mpl.rcParams['axes.unicode_minus'] = False # ป้องกันปัญหาเครื่องหมายลบแสดงเป็นสี่เหลี่ยม

print("Library and font setup complete")"""
))

# 3. Data Loading
cells.append(nbf.v4.new_markdown_cell(
r"""## 3. Data Loading & Exploratory Data Analysis (EDA)
Read the data from the CSV file and plot scatter plots to observe the relationship between $x$ and $y$."""
))
cells.append(nbf.v4.new_code_cell(
r"""# อ่านข้อมูล
df = pd.read_csv('CPE342_Assignment 2_Data.csv')
X = df['X'].values
Y = df['Y'].values

# แสดงข้อมูลเบื้องต้น
display(df.head())

# พล็อตกราฟเพื่อดูแนวโน้ม
plt.figure(figsize=(12, 5))

# กราฟ X กับ Y
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', alpha=0.6, edgecolor='k')
plt.title('Scatter plot of X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)

# กราฟ X^2 กับ Y
plt.subplot(1, 2, 2)
plt.scatter(X**2, Y, color='green', alpha=0.6, edgecolor='k')
plt.title('Scatter plot of $X^2$ vs Y')
plt.xlabel('$X^2$')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""
))

# 4. Task 1
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 1 — Loss Function
Given our model $\hat{y}_i = C_0 + C_1 x_i^2$, 

The loss function we use is the **Mean Squared Error (MSE)**, which computes the average squared differences (residuals) across all data points:

$$ L(C_0, C_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

Substituting $\hat{y}_i$ gives our target equation:
$$ L(C_0, C_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2)^2 $$"""
))

# 5. Task 2
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 2 — Gradient of Each Coefficient
To use Gradient Descent, we need to compute the partial derivatives of the loss function $L$ with respect to each parameter. 

Let $e_i = y_i - C_0 - C_1 x_i^2$ be the residual error.

**1. Gradient with respect to $C_0$:**
$$ \frac{\partial L}{\partial C_0} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - C_0 - C_1 x_i^2) \cdot \frac{\partial}{\partial C_0}(-C_0) $$
$$ \frac{\partial L}{\partial C_0} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2) $$

**2. Gradient with respect to $C_1$:**
$$ \frac{\partial L}{\partial C_1} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - C_0 - C_1 x_i^2) \cdot \frac{\partial}{\partial C_1}(-C_1 x_i^2) $$
$$ \frac{\partial L}{\partial C_1} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2) x_i^2 $$"""
))

# 6. Task 3
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 3 — Update Rules
In each iteration $t$, the parameters are updated in the opposite direction of the gradient to minimize the loss, scaled by the learning rate $\eta$:

$$ C_0^{(t+1)} = C_0^{(t)} - \eta \cdot \frac{\partial L}{\partial C_0} $$
$$ C_1^{(t+1)} = C_1^{(t)} - \eta \cdot \frac{\partial L}{\partial C_1} $$"""
))

# 7. Task 4 (Code)
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 4 — Gradient Descent Implementation
In this section, we implement the code to find $C_0$ and $C_1$ using basic NumPy, based on the mathematical equations derived in the previous tasks."""
))
cells.append(nbf.v4.new_code_cell(
r"""# Hyperparameters
learning_rate = 0.01  
n_iterations = 5000   
n = len(Y)            

# Initialize parameters
C0 = 0.0
C1 = 0.0

# Prepare lists to store history (for visualization later)
history_loss = []
history_C0 = []
history_C1 = []

# Pre-compute X^2 to reduce computation inside the loop
X_sq = X**2

# Gradient Descent Loop
for i in range(n_iterations):
    # 1. Calculate Predictions
    Y_pred = C0 + C1 * X_sq
    
    # 2. Calculate Errors/Residuals
    error = Y - Y_pred
    
    # 3. Calculate Loss (MSE)
    loss = np.mean(error**2)
    history_loss.append(loss)
    history_C0.append(C0)
    history_C1.append(C1)
    
    # 4. Calculate Gradients
    grad_C0 = -(2/n) * np.sum(error)
    grad_C1 = -(2/n) * np.sum(error * X_sq)
    
    # 5. Update Parameters
    C0 = C0 - learning_rate * grad_C0
    C1 = C1 - learning_rate * grad_C1
    
    # Print progress every 500 iterations
    if i % 500 == 0 or i == n_iterations - 1:
        print(f"Iteration {i:4d} | Loss: {loss:.5f} | C0: {C0:.5f} | C1: {C1:.5f}")

print("-" * 50)
print(f"Training complete over {n_iterations} iterations")
print(f"Optimized Parameters: C0 = {C0:.4f}, C1 = {C1:.4f}")
print(f"Final MSE Loss: {history_loss[-1]:.6f}")"""
))

# 8. Task 5 (Visualizations)
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 5 — ผลลัพธ์และการแสดงภาพ (Results & Visualizations)
เราจะมาดูกันว่าโมเดลที่เราเทรนมามีพฤติกรรมอย่างไร และฟิตกับข้อมูลได้ดีแค่ไหน"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 1: การลดลงของ Loss (Learning Curve)
plt.figure(figsize=(8, 5))
plt.plot(range(n_iterations), history_loss, color='red', linewidth=2)
plt.title('Learning Curve (MSE Loss per Iteration)')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 2: เส้นทางการลู่เข้าของพารามิเตอร์ C0 และ C1
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), history_C0, color='blue', linewidth=2)
plt.title('Convergence of C0 (Intercept)')
plt.xlabel('Iteration')
plt.ylabel('C0')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), history_C1, color='orange', linewidth=2)
plt.title('Convergence of C1 (Coefficient of $X^2$)')
plt.xlabel('Iteration')
plt.ylabel('C1')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 3: เส้นโค้งที่ฟิตแล้วเทียบกับจุดข้อมูลจริง (Fitted Curve)
plt.figure(figsize=(9, 6))

# จุดข้อมูลจริง
plt.scatter(X, Y, color='black', alpha=0.5, label='Actual Data', edgecolor='k')

# สร้างจุด x สำหรับวาดเส้นโค้งให้เรียบเนียน
x_line = np.linspace(min(X), max(X), 100)
y_line = C0 + C1 * (x_line**2)

plt.plot(x_line, y_line, color='red', linewidth=3, label=f'Model: $\hat{{y}} = {C0:.4f} + {C1:.4f}x^2$')

plt.title('Quadratic Model Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 4: ความคลาดเคลื่อน (Residuals) แต่ละจุด
Y_pred_final = C0 + C1 * X_sq
residuals = Y - Y_pred_final

plt.figure(figsize=(9, 5))
plt.scatter(X, residuals, color='purple', alpha=0.7, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals of the Model')
plt.xlabel('X')
plt.ylabel('Residual ($y_i - \hat{y}_i$)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""
))

nb['cells'] = cells

with open('Assignment_2_GD.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
