import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# 1. Title & Problem Overview
cells.append(nbf.v4.new_markdown_cell(
r"""# CPE 342 Machine Learning — Assignment 2: Training Models
**ผู้จัดทำ:** 67070501042 วิศิษฐ์ สุวรรณเนาว์

## 1. ภาพรวมของปัญหา (Problem Overview)
ในงานนี้เราจะใช้ **วิธีการวนซ้ำ (Iterative approach)** หรือที่เรียกว่า **Gradient Descent (GD)** เพื่อฝึกสอนโมเดล (Train model) ให้ฟิตเข้ากับชุดข้อมูลที่มี 100 จุดสังเกต ($n=100$) 

เป้าหมายคือการสร้างโมเดลสมการพหุนามกำลังสอง (Quadratic model) ที่ไม่มีพจน์ $x$ เชิงเส้น:
$$ \hat{y} = C_0 + C_1 x^2 $$
โดยที่:
- $C_0$ คือ จุดตัดแกน Y (Intercept หรือ Bias)
- $C_1$ คือ สัมประสิทธิ์ของตัวแปร $x^2$ (Slope of $x^2$)"""
))

# 2. Font & Library Setup
cells.append(nbf.v4.new_markdown_cell(
r"""## 2. การตั้งค่าไลบรารีและฟอนต์ (Setup)
ทำการโหลดฟอนต์ Sarabun เพื่อให้กราฟสามารถแสดงผลภาษาไทยได้ และนำเข้าไลบรารีที่จำเป็น"""
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

print("ตั้งค่าไลบรารีและฟอนต์สำเร็จ")"""
))

# 3. Data Loading
cells.append(nbf.v4.new_markdown_cell(
r"""## 3. นำเข้าและสำรวจข้อมูล (Data Loading & EDA)
อ่านข้อมูลจากไฟล์ CSV และพล็อตกราฟกระจายตัว (Scatter plot) เพื่อดูความสัมพันธ์ระหว่าง $x$ และ $y$"""
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
plt.title('การกระจายตัวของข้อมูล: X กับ Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)

# กราฟ X^2 กับ Y
plt.subplot(1, 2, 2)
plt.scatter(X**2, Y, color='green', alpha=0.6, edgecolor='k')
plt.title('การกระจายตัวของข้อมูล: $X^2$ กับ Y')
plt.xlabel('$X^2$')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""
))

# 4. Task 1
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 1 — ฟังก์ชันความสูญเสีย (Loss Function)
กำหนดให้โมเดลของเราคือ $\hat{y}_i = C_0 + C_1 x_i^2$ 

ฟังก์ชันความสูญเสียที่เราใช้คือ **Mean Squared Error (MSE)** ซึ่งคำนวณจากค่าเฉลี่ยของกำลังสองของความคลาดเคลื่อน (Residuals) ของทุกๆ ข้อมูล:

$$ L(C_0, C_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

เมื่อแทนค่า $\hat{y}_i$ จะได้สมการเป้าหมายของเรา:
$$ L(C_0, C_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2)^2 $$"""
))

# 5. Task 2
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 2 — การหาอนุพันธ์ย่อย (Gradient of Each Coefficient)
เพื่อที่จะใช้ Gradient Descent เราต้องหาอนุพันธ์ย่อย (Partial derivatives) ของ $L$ เทียบกับพารามิเตอร์แต่ละตัว 

ให้ $e_i = y_i - C_0 - C_1 x_i^2$ เป็นค่าความคลาดเคลื่อน

**1. Gradient เทียบกับ $C_0$:**
$$ \frac{\partial L}{\partial C_0} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - C_0 - C_1 x_i^2) \cdot \frac{\partial}{\partial C_0}(-C_0) $$
$$ \frac{\partial L}{\partial C_0} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2) $$

**2. Gradient เทียบกับ $C_1$:**
$$ \frac{\partial L}{\partial C_1} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - C_0 - C_1 x_i^2) \cdot \frac{\partial}{\partial C_1}(-C_1 x_i^2) $$
$$ \frac{\partial L}{\partial C_1} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - C_0 - C_1 x_i^2) x_i^2 $$"""
))

# 6. Task 3
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 3 — กฎการอัปเดตพารามิเตอร์ (Update Rules)
ในการวนซ้ำแต่ละรอบ $t$ พารามิเตอร์จะถูกปรับค่าในทิศทางที่ตรงข้ามกับ Gradient เพื่อทำให้ค่า Loss ลดลง โดยมี $\eta$ (Learning Rate) เป็นตัวควบคุมขนาดของก้าว (Step size):

$$ C_0^{(t+1)} = C_0^{(t)} - \eta \cdot \frac{\partial L}{\partial C_0} $$
$$ C_1^{(t+1)} = C_1^{(t)} - \eta \cdot \frac{\partial L}{\partial C_1} $$"""
))

# 7. Task 4 (Code)
cells.append(nbf.v4.new_markdown_cell(
r"""## Task 4 — การเขียนโปรแกรม Gradient Descent (Implementation)
ในส่วนนี้จะเป็นการเขียนโค้ดเพื่อหาค่า $C_0$ และ $C_1$ โดยใช้ NumPy พื้นฐาน ตามสมการคณิตศาสตร์ที่หามาได้ใน Task ก่อนหน้า"""
))
cells.append(nbf.v4.new_code_cell(
r"""# Hyperparameters การตั้งค่าเบื้องต้น
learning_rate = 0.01  # อัตราการเรียนรู้
n_iterations = 5000   # จำนวนรอบการวนซ้ำ
n = len(Y)            # จำนวนจุดข้อมูล

# กำหนดค่าเริ่มต้นของพารามิเตอร์
C0 = 0.0
C1 = 0.0

# สร้าง list สำหรับเก็บประวัติการทำงานเพื่อนำไปพล็อตกราฟ
history_loss = []
history_C0 = []
history_C1 = []

# ตัวแปร X^2 เพื่อความสะดวกรวดเร็วในการคำนวณ
X_sq = X**2

# เริ่มวงลูป Gradient Descent
for i in range(n_iterations):
    # 1. คำนวณค่าพยากรณ์ (Predictions)
    Y_pred = C0 + C1 * X_sq
    
    # 2. คำนวณความคลาดเคลื่อน (Errors/Residuals)
    error = Y - Y_pred
    
    # 3. คำนวณค่า Loss (MSE)
    loss = np.mean(error**2)
    history_loss.append(loss)
    history_C0.append(C0)
    history_C1.append(C1)
    
    # 4. คำนวณ Gradients (ตามสมการที่หามา)
    grad_C0 = -(2/n) * np.sum(error)
    grad_C1 = -(2/n) * np.sum(error * X_sq)
    
    # 5. อัปเดตพารามิเตอร์
    C0 = C0 - learning_rate * grad_C0
    C1 = C1 - learning_rate * grad_C1
    
    # พิมพ์ผลลัพธ์ทุกๆ 500 รอบ
    if i % 500 == 0 or i == n_iterations - 1:
        print(f"Iteration {i:4d} | Loss: {loss:.5f} | C0: {C0:.5f} | C1: {C1:.5f}")

print("-" * 50)
print(f"Final Parameters: C0 = {C0:.5f}, C1 = {C1:.5f}")
print(f"Final MSE Loss: {loss:.5f}")"""
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
plt.title('การเปลี่ยนแปลงของ MSE Loss ตลอดการวนซ้ำ')
plt.xlabel('รอบการวนซ้ำ (Iteration)')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 2: เส้นทางการลู่เข้าของพารามิเตอร์ C0 และ C1
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), history_C0, color='blue', linewidth=2)
plt.title('การลู่เข้าของ C0 (Intercept)')
plt.xlabel('รอบการวนซ้ำ')
plt.ylabel('C0')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), history_C1, color='orange', linewidth=2)
plt.title('การลู่เข้าของ C1 (Coefficient ของ $X^2$)')
plt.xlabel('รอบการวนซ้ำ')
plt.ylabel('C1')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""
))
cells.append(nbf.v4.new_code_cell(
r"""# กราฟที่ 3: เส้นโค้งที่ฟิตแล้วเทียบกับจุดข้อมูลจริง (Fitted Curve)
plt.figure(figsize=(9, 6))

# จุดข้อมูลจริง
plt.scatter(X, Y, color='black', alpha=0.5, label='ข้อมูลจริง (Data)', edgecolor='k')

# สร้างจุด x สำหรับวาดเส้นโค้งให้เรียบเนียน
x_line = np.linspace(min(X), max(X), 100)
y_line = C0 + C1 * (x_line**2)

plt.plot(x_line, y_line, color='red', linewidth=3, label=f'โมเดล: $\hat{{y}} = {C0:.4f} {C1:.4f}x^2$')

plt.title('การฟิตโมเดล Quadratic เข้ากับข้อมูล')
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
plt.title('ความคลาดเคลื่อน (Residuals) แบ่งตามจุดข้อมูล X')
plt.xlabel('X')
plt.ylabel('Residual ($y_i - \hat{y}_i$)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""
))

nb['cells'] = cells

with open('Assignment_2_GD.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
