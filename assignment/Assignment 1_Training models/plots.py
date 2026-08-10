"""
plots.py — สร้างกราฟสำหรับ Assignment 1: OLS Regression
ใช้ฟอนต์ Sarabun ในกราฟ (load จาก .ttf โดยตรง)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ===== โหลดฟอนต์ Sarabun จาก path ตรง ๆ =====
font_dir = os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'Windows', 'Fonts')
sarabun_regular = os.path.join(font_dir, 'Sarabun-Regular.ttf')
sarabun_bold = os.path.join(font_dir, 'Sarabun-Bold.ttf')

prop_regular = fm.FontProperties(fname=sarabun_regular, size=12)
prop_bold = fm.FontProperties(fname=sarabun_bold, size=12)
prop_title = fm.FontProperties(fname=sarabun_bold, size=14)
prop_legend = fm.FontProperties(fname=sarabun_regular, size=10)
prop_annot = fm.FontProperties(fname=sarabun_regular, size=9)

# ===== ข้อมูล =====
X = np.array([3.0, 5.0, 2.0, 7.0, 8.0, 1.0, 4.0, 6.0, 9.0, 10.0])
Y = np.array([6.0, 9.0, 4.0, 10.0, 12.0, 3.0, 7.0, 8.0, 13.0, 15.0])
n = len(X)

# ===== คำนวณ OLS =====
sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XY = np.sum(X * Y)
sum_X2 = np.sum(X**2)
X_bar = np.mean(X)
Y_bar = np.mean(Y)

beta = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)
alpha = Y_bar - beta * X_bar

Y_hat = alpha + beta * X
residuals = Y - Y_hat

SS_total = np.sum((Y - Y_bar)**2)
SS_residual = np.sum(residuals**2)
R_squared = 1 - SS_residual / SS_total

X_pred = 12.0
Y_pred = alpha + beta * X_pred

print(f"beta = {beta:.4f}, alpha = {alpha:.4f}")
print(f"R^2 = {R_squared:.4f}")
print(f"Prediction X=12: Y_hat = {Y_pred:.4f}")

# ===== กราฟที่ 1: Scatter plot + Regression line =====
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(X, Y, color='#2563EB', s=90, zorder=5, edgecolors='white', linewidths=1.5,
           label='Observed data')

X_line = np.linspace(0, 13, 200)
Y_line = alpha + beta * X_line
ax.plot(X_line, Y_line, color='#DC2626', linewidth=2.2,
        label=f'$\\hat{{Y}} = {alpha:.4f} + {beta:.4f}X$', zorder=4)

ax.scatter([12], [Y_pred], color='#16A34A', s=120, zorder=6, marker='D',
           edgecolors='white', linewidths=1.5,
           label=f'Prediction at X=12 (Y={Y_pred:.2f})')
ax.plot([12, 12], [0, Y_pred], color='#16A34A', linestyle='--', linewidth=1, alpha=0.6, zorder=3)
ax.plot([0, 12], [Y_pred, Y_pred], color='#16A34A', linestyle='--', linewidth=1, alpha=0.6, zorder=3)

for i in range(n):
    ax.annotate(f'({X[i]:.0f}, {Y[i]:.0f})', (X[i], Y[i]),
                textcoords="offset points", xytext=(8, 8),
                fontproperties=prop_annot, color='#374151')

ax.set_xlabel('Advertising Budget (X) [$1,000s]', fontproperties=prop_bold)
ax.set_ylabel('Sales (Y) [1,000 units]', fontproperties=prop_bold)
ax.set_title('OLS Linear Regression: Advertising Budget vs Sales',
             fontproperties=prop_title, pad=15)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9, prop=prop_legend)
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(0, 18)
ax.grid(True, alpha=0.3)

# ตั้ง tick labels ให้ใช้ Sarabun
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop_regular)

plt.tight_layout()
fig.savefig('scatter_regression.pdf', dpi=300, bbox_inches='tight')
fig.savefig('scatter_regression.png', dpi=200, bbox_inches='tight')
print("Saved scatter_regression.pdf / .png")
plt.close()

# ===== กราฟที่ 2: Residual Plot =====
fig2, ax2 = plt.subplots(figsize=(8, 4.5))

colors = ['#DC2626' if r < 0 else '#2563EB' for r in residuals]
ax2.bar(X, residuals, width=0.35, color=colors, edgecolor='white', linewidth=0.8, alpha=0.85, zorder=4)
ax2.axhline(y=0, color='#374151', linewidth=1.5, zorder=3)

for i in range(n):
    offset_y = 0.08 if residuals[i] >= 0 else -0.08
    va = 'bottom' if residuals[i] >= 0 else 'top'
    ax2.annotate(f'{residuals[i]:.2f}', (X[i], residuals[i] + offset_y),
                 ha='center', va=va, fontproperties=fm.FontProperties(fname=sarabun_bold, size=9),
                 color=colors[i])

ax2.set_xlabel('Advertising Budget (X) [$1,000s]', fontproperties=prop_bold)
ax2.set_ylabel('Residual ($e_i = Y_i - \\hat{Y}_i$)', fontproperties=prop_bold)
ax2.set_title('Residual Plot', fontproperties=prop_title, pad=15)
ax2.set_xlim(0, 11)
ax2.grid(True, alpha=0.3, axis='y')

for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontproperties(prop_regular)

plt.tight_layout()
fig2.savefig('residual_plot.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('residual_plot.png', dpi=200, bbox_inches='tight')
print("Saved residual_plot.pdf / .png")
plt.close()

print("\nAll plots generated with Sarabun font!")
