import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import scipy.stats as stats

# --- 1. Robust Sarabun Font Setup ---
font_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')
sarabun_r_path = os.path.join(font_dir, 'Sarabun-Regular.ttf')
sarabun_b_path = os.path.join(font_dir, 'Sarabun-Bold.ttf')

prop_r = fm.FontProperties(fname=sarabun_r_path, size=11)
prop_b = fm.FontProperties(fname=sarabun_b_path, size=12)
prop_title = fm.FontProperties(fname=sarabun_b_path, size=14)
prop_small = fm.FontProperties(fname=sarabun_r_path, size=9)

def apply_font(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(prop_r)

plt.style.use('seaborn-v0_8-whitegrid')

# --- 2. Data & OLS Calculation ---
X = np.array([3., 5., 2., 7., 8., 1., 4., 6., 9., 10.])
Y = np.array([6., 9., 4., 10., 12., 3., 7., 8., 13., 15.])
n = len(X)
months = np.arange(1, n + 1)

X_bar, Y_bar = X.mean(), Y.mean()
Sxx = np.sum((X - X_bar)**2)
Sxy = np.sum((X - X_bar)*(Y - Y_bar))
beta = Sxy / Sxx
alpha = Y_bar - beta * X_bar

Y_hat = alpha + beta * X
residuals = Y - Y_hat

def avoid_overlap(x, y, x_offset=0, y_offset=0):
    return (x_offset, y_offset)

# --- Plot 1: Scatter + Regression Line ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(X, Y, color='#4F46E5', s=100, edgecolors='white', linewidths=1.5, label='Actual Data')
X_line = np.linspace(0, 13, 100)
ax1.plot(X_line, alpha + beta * X_line, color='#E11D48', linewidth=2.5, label=fr'Regression Line: Y = {alpha:.2f} + {beta:.2f}X')
ax1.scatter([12], [alpha + beta * 12], color='#10B981', s=120, marker='D', edgecolors='white', linewidths=1.5, label='Prediction at X=12')

for i in range(n):
    ax1.annotate(f'({X[i]:.0f}, {Y[i]:.0f})', (X[i], Y[i]), textcoords='offset points', xytext=(8, -12), fontproperties=prop_small, color='#4B5563', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

ax1.set_xlabel('Advertising Budget (X) [$1,000s]', fontproperties=prop_b)
ax1.set_ylabel('Sales (Y) [1,000 units]', fontproperties=prop_b)
ax1.set_title('1. Advertising Budget vs Sales', fontproperties=prop_title)
legend = ax1.legend(prop=prop_r)
apply_font(ax1)
ax1.set_xlim(-0.5, 13.5)
ax1.set_ylim(0, 18)
plt.tight_layout()
fig1.savefig('plot_1_regression.pdf', dpi=300)
plt.close()

# --- Plot 2: Residuals Lollipop Chart ---
fig2, ax2 = plt.subplots(figsize=(8, 4))
markerline, stemlines, baseline = ax2.stem(X, residuals, linefmt='-', markerfmt='o', basefmt='k-')
plt.setp(markerline, color='#0284C7', markersize=8, markeredgecolor='white', markeredgewidth=1)
plt.setp(stemlines, color='#0284C7', linewidth=2)
plt.setp(baseline, color='#1F2937', linewidth=1.5)

for i in range(n):
    offset = 0.15 if residuals[i] > 0 else -0.35
    ax2.text(X[i], residuals[i] + offset, f'{residuals[i]:.2f}', ha='center', fontproperties=prop_small, color='#0284C7', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

ax2.set_xlabel('Advertising Budget (X) [$1,000s]', fontproperties=prop_b)
ax2.set_ylabel('Residual Error ($e_i$)', fontproperties=prop_b)
ax2.set_title('2. Residuals (Errors) per Observation', fontproperties=prop_title)
apply_font(ax2)
ax2.set_xlim(-0.5, 12.5)
ax2.set_ylim(min(residuals) - 0.6, max(residuals) + 0.6)
plt.tight_layout()
fig2.savefig('plot_2_residuals_lollipop.pdf', dpi=300)
plt.close()

# --- Plot 3: Residuals vs Fitted ---
fig3, ax3 = plt.subplots(figsize=(6, 4.5))
ax3.scatter(Y_hat, residuals, color='#F59E0B', s=90, edgecolors='white', linewidths=1.2)
ax3.axhline(0, color='#1F2937', linestyle='--', linewidth=1.5)
for i in range(n):
    ax3.annotate(f'M{i+1}', (Y_hat[i], residuals[i]), textcoords='offset points', xytext=(6, 5), fontproperties=prop_small, color='#4B5563')
ax3.set_xlabel(r'Fitted Values ($\hat{Y}$)', fontproperties=prop_b)
ax3.set_ylabel('Residuals', fontproperties=prop_b)
ax3.set_title('3. Residuals vs Fitted Values', fontproperties=prop_title)
apply_font(ax3)
plt.tight_layout()
fig3.savefig('plot_3_res_vs_fitted.pdf', dpi=300)
plt.close()

# --- Plot 4: Residual Distribution (Histogram) ---
fig4, ax4 = plt.subplots(figsize=(6, 4.5))
ax4.hist(residuals, bins=6, color='#8B5CF6', edgecolor='white', alpha=0.8, density=True)
x_range = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 100)
ax4.plot(x_range, stats.norm.pdf(x_range, 0, residuals.std(ddof=1)), color='#E11D48', linewidth=2)
ax4.axvline(0, color='#1F2937', linestyle='--', linewidth=1.2)
ax4.set_xlabel('Residual Value', fontproperties=prop_b)
ax4.set_ylabel('Density', fontproperties=prop_b)
ax4.set_title('4. Residuals Distribution', fontproperties=prop_title)
apply_font(ax4)
plt.tight_layout()
fig4.savefig('plot_4_res_dist.pdf', dpi=300)
plt.close()

# --- Plot 5: Normal Q-Q Plot ---
fig5, ax5 = plt.subplots(figsize=(6, 4.5))
(osm, osr), (slope_qq, intercept_qq, _) = stats.probplot(residuals, dist='norm')
ax5.scatter(osm, osr, color='#EC4899', s=90, edgecolors='white', linewidths=1.2)
xq = np.array([osm.min(), osm.max()])
ax5.plot(xq, slope_qq * xq + intercept_qq, color='#1F2937', linewidth=2)
ax5.set_xlabel('Theoretical Quantiles', fontproperties=prop_b)
ax5.set_ylabel('Sample Quantiles', fontproperties=prop_b)
ax5.set_title('5. Normal Q-Q Plot', fontproperties=prop_title)
apply_font(ax5)
plt.tight_layout()
fig5.savefig('plot_5_qq.pdf', dpi=300)
plt.close()

# --- Plot 6: Scale-Location Plot ---
fig6, ax6 = plt.subplots(figsize=(6, 4.5))
sqrt_abs_res = np.sqrt(np.abs(residuals))
ax6.scatter(Y_hat, sqrt_abs_res, color='#10B981', s=90, edgecolors='white', linewidths=1.2)
for i in range(n):
    ax6.annotate(f'M{i+1}', (Y_hat[i], sqrt_abs_res[i]), textcoords='offset points', xytext=(6, 5), fontproperties=prop_small, color='#4B5563')
ax6.set_xlabel(r'Fitted Values ($\hat{Y}$)', fontproperties=prop_b)
ax6.set_ylabel(r'$\sqrt{|Residuals|}$', fontproperties=prop_b)
ax6.set_title('6. Scale-Location Plot', fontproperties=prop_title)
apply_font(ax6)
plt.tight_layout()
fig6.savefig('plot_6_scale_loc.pdf', dpi=300)
plt.close()

# --- Plot 7: Actual vs Predicted ---
fig7, ax7 = plt.subplots(figsize=(6, 6))
sc = ax7.scatter(Y_hat, Y, c=months, cmap='plasma', s=120, edgecolors='white', linewidths=1.5)
cbar = fig7.colorbar(sc, ax=ax7)
cbar.set_label('Month', fontproperties=prop_b)
cbar.ax.tick_params(labelsize=11)
for t in cbar.ax.get_yticklabels():
    t.set_fontproperties(prop_r)

lims = [min(Y.min(), Y_hat.min()) - 1, max(Y.max(), Y_hat.max()) + 1]
ax7.plot(lims, lims, color='#1F2937', linestyle='--', linewidth=2, label='Perfect Fit')

for i in range(n):
    ax7.annotate(f'M{i+1}', (Y_hat[i], Y[i]), textcoords='offset points', xytext=(-15, 8), fontproperties=prop_small, color='#4B5563', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

ax7.set_xlabel(r'Predicted Sales ($\hat{Y}$)', fontproperties=prop_b)
ax7.set_ylabel('Actual Sales ($Y$)', fontproperties=prop_b)
ax7.set_title('7. Actual vs Predicted Sales', fontproperties=prop_title)
ax7.legend(prop=prop_r)
apply_font(ax7)
ax7.set_xlim(lims)
ax7.set_ylim(lims)
ax7.set_aspect('equal')
plt.tight_layout()
fig7.savefig('plot_7_actual_vs_pred.pdf', dpi=300)
plt.close()

print("7 individual plots generated successfully using strict Sarabun FontProperties.")
