import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Create a rich synthetic dataset ────────────────────────────────────────
np.random.seed(42)
n = 500

regions   = np.random.choice(['North', 'South', 'East', 'West'], n)
products  = np.random.choice(['Electronics', 'Clothing', 'Food', 'Furniture'], n)
months    = np.random.choice(range(1, 13), n)
units_sold = np.random.randint(10, 300, n)
unit_price = np.round(np.random.uniform(5, 500, n), 2)
discount   = np.round(np.random.uniform(0, 0.4, n), 2)
revenue    = np.round(units_sold * unit_price * (1 - discount), 2)
satisfaction = np.clip(np.random.normal(3.8, 0.8, n), 1, 5).round(1)
returns    = np.where(satisfaction < 2.5, 1, 0)

df = pd.DataFrame({
    'region': regions, 'product': products, 'month': months,
    'units_sold': units_sold, 'unit_price': unit_price,
    'discount': discount, 'revenue': revenue,
    'satisfaction': satisfaction, 'returns': returns
})

print("=== DATASET OVERVIEW ===")
print(df.head(10).to_string())
print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")

# ── 2. Descriptive Statistics ──────────────────────────────────────────────────
print("\n=== 1. DESCRIPTIVE STATISTICS ===")
desc = df[['units_sold','unit_price','discount','revenue','satisfaction']].describe().round(2)
print(desc)

# ── 3. GroupBy Aggregation ─────────────────────────────────────────────────────
print("\n=== 2. GROUPBY AGGREGATION (Revenue by Region & Product) ===")
grp = df.groupby(['region','product'])['revenue'].agg(['sum','mean','count']).round(2)
print(grp.to_string())

# ── 4. Correlation Matrix ──────────────────────────────────────────────────────
print("\n=== 3. CORRELATION MATRIX ===")
corr = df[['units_sold','unit_price','discount','revenue','satisfaction']].corr().round(3)
print(corr)

# ── 5. Pivot Table ─────────────────────────────────────────────────────────────
print("\n=== 4. PIVOT TABLE (Avg Revenue by Region × Product) ===")
pivot = df.pivot_table(values='revenue', index='region',
                       columns='product', aggfunc='mean').round(0)
print(pivot)

# ── 6. Feature Engineering ─────────────────────────────────────────────────────
print("\n=== 5. FEATURE ENGINEERING ===")
df['revenue_per_unit'] = (df['revenue'] / df['units_sold']).round(2)
df['high_value']       = (df['revenue'] > df['revenue'].median()).astype(int)
df['season']           = pd.cut(df['month'], bins=[0,3,6,9,12],
                                labels=['Winter','Spring','Summer','Fall'])
df['discount_tier']    = pd.cut(df['discount'], bins=[-0.01,0.1,0.25,0.4],
                                labels=['Low','Mid','High'])
print(df[['revenue_per_unit','high_value','season','discount_tier']].head(10))

# ── 7. Standardisation ────────────────────────────────────────────────────────
print("\n=== 6. STANDARDISATION (Z-score) ===")
scaler = StandardScaler()
cols_to_scale = ['units_sold','unit_price','revenue']
scaled = pd.DataFrame(scaler.fit_transform(df[cols_to_scale]),
                       columns=[c+'_z' for c in cols_to_scale])
print(scaled.describe().round(3))

# ── 8. NumPy Array Conversion & np.where ──────────────────────────────────────
print("\n=== 7. NUMPY CONVERSION & np.where ===")
revenue_np = df['revenue'].to_numpy()
print(f"Type  : {type(revenue_np)}")
print(f"Shape : {revenue_np.shape}")
print(f"Dtype : {revenue_np.dtype}")
print(f"Mean  : {revenue_np.mean():.2f}  |  Std: {revenue_np.std():.2f}")
print(f"Min   : {revenue_np.min():.2f}  |  Max: {revenue_np.max():.2f}")

threshold    = np.percentile(revenue_np, 75)
revenue_cat  = np.where(revenue_np >= threshold, 'Premium', 'Standard')
premium_mask = np.where(revenue_np >= threshold, True, False)
df['revenue_category'] = revenue_cat

print(f"\nnp.where threshold (75th percentile): {threshold:.2f}")
print(f"Premium  count : {(revenue_cat=='Premium').sum()}")
print(f"Standard count : {(revenue_cat=='Standard').sum()}")
print(f"\nFirst 15 values  : {revenue_np[:15]}")
print(f"First 15 labels  : {revenue_cat[:15]}")

# ── TOP-LEVEL STATS for dashboard label ───────────────────────────────────────
total_rev   = df['revenue'].sum()
avg_sat     = df['satisfaction'].mean()
return_rate = df['returns'].mean() * 100
top_product = df.groupby('product')['revenue'].sum().idxmax()
top_region  = df.groupby('region')['revenue'].sum().idxmax()

print(f"\n=== SUMMARY METRICS ===")
print(f"Total Revenue   : ${total_rev:,.0f}")
print(f"Avg Satisfaction: {avg_sat:.2f}")
print(f"Return Rate     : {return_rate:.1f}%")
print(f"Top Product     : {top_product}")
print(f"Top Region      : {top_region}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
DARK   = '#0d1117'
PANEL  = '#161b22'
ACC1   = '#58a6ff'   # blue
ACC2   = '#3fb950'   # green
ACC3   = '#f78166'   # red-orange
ACC4   = '#d2a8ff'   # purple
ACC5   = '#ffa657'   # amber
ACCENT_LIST = [ACC1, ACC2, ACC3, ACC4]

plt.rcParams.update({
    'figure.facecolor'  : DARK,
    'axes.facecolor'    : PANEL,
    'axes.edgecolor'    : '#30363d',
    'axes.labelcolor'   : '#c9d1d9',
    'xtick.color'       : '#8b949e',
    'ytick.color'       : '#8b949e',
    'text.color'        : '#c9d1d9',
    'grid.color'        : '#21262d',
    'grid.linewidth'    : 0.6,
    'font.family'       : 'monospace',
})

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

# ── TITLE BANNER ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, '  RETAIL SALES ANALYTICS DASHBOARD',
         ha='center', va='top', fontsize=19, fontweight='bold',
         color='#ffffff', fontfamily='monospace')
fig.text(0.5, 0.945, f'n = {n} transactions  |  6 analysis methods  |  numpy.where demonstrated',
         ha='center', va='top', fontsize=9, color='#8b949e')

# KPI boxes ───────────────────────────────────────────────────────────────────
kpis = [
    (f"${total_rev/1e6:.2f}M", "Total Revenue"),
    (f"{avg_sat:.2f} / 5", "Avg Satisfaction"),
    (f"{return_rate:.1f}%", "Return Rate"),
    (top_product, "Top Product"),
    (top_region, "Top Region"),
]
kpi_ax = fig.add_axes([0.01, 0.905, 0.98, 0.032])
kpi_ax.set_facecolor(DARK); kpi_ax.axis('off')
for i, (val, lbl) in enumerate(kpis):
    x = 0.1 + i * 0.2
    kpi_ax.text(x, 0.85, val, ha='center', va='top',
                fontsize=11, fontweight='bold', color=ACCENT_LIST[i % 4])
    kpi_ax.text(x, 0.15, lbl, ha='center', va='bottom',
                fontsize=7.5, color='#8b949e')

# ── PLOT 1: Revenue by Region (Bar) ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
reg_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
bars = ax1.bar(reg_rev.index, reg_rev.values / 1e6, color=ACCENT_LIST,
               edgecolor='#21262d', linewidth=0.6, zorder=3)
ax1.set_title('Revenue by Region', fontsize=11, fontweight='bold', pad=8, color='#ffffff')
ax1.set_ylabel('Revenue ($M)', fontsize=8)
ax1.grid(axis='y', zorder=0)
ax1.set_ylim(0, reg_rev.max() / 1e6 * 1.2)
for bar, val in zip(bars, reg_rev.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'${val/1e6:.2f}M', ha='center', va='bottom', fontsize=7.5,
             color='#ffffff', fontweight='bold')

# ── PLOT 2: Revenue Distribution – Premium vs Standard (Violin) ───────────────
ax2 = fig.add_subplot(gs[0, 1])
grp_data = [df.loc[df['revenue_category']=='Premium','revenue'].values,
            df.loc[df['revenue_category']=='Standard','revenue'].values]
parts = ax2.violinplot(grp_data, positions=[1, 2], showmedians=True,
                       showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([ACC1, ACC3][i])
    pc.set_alpha(0.65)
parts['cmedians'].set_color('#ffffff'); parts['cmedians'].set_linewidth(1.5)
parts['cmins'].set_color('#8b949e'); parts['cmaxes'].set_color('#8b949e')
parts['cbars'].set_color('#8b949e')
ax2.set_xticks([1, 2]); ax2.set_xticklabels(['Premium', 'Standard'])
ax2.set_title('np.where: Premium vs Standard\nRevenue Distribution', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax2.set_ylabel('Revenue ($)', fontsize=8)
ax2.grid(axis='y')
note = f'Threshold: ${threshold:,.0f} (75th pct)'
ax2.text(0.97, 0.97, note, transform=ax2.transAxes, ha='right', va='top',
         fontsize=7, color=ACC5, style='italic')

# ── PLOT 3: Correlation Heatmap ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
corr_cols = ['units_sold','unit_price','discount','revenue','satisfaction']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, ax=ax3, cmap=cmap, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 7.5},
            linewidths=0.5, linecolor='#0d1117',
            cbar_kws={'shrink': 0.8})
ax3.set_title('Correlation Matrix', fontsize=11, fontweight='bold', pad=8, color='#ffffff')
ax3.tick_params(axis='x', rotation=30, labelsize=7.5)
ax3.tick_params(axis='y', rotation=0, labelsize=7.5)

# ── PLOT 4: Monthly Revenue Trend (Line) ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
monthly = df.groupby(['month','product'])['revenue'].sum().unstack()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors_p = [ACC1, ACC2, ACC3, ACC4]
for col, col_color in zip(monthly.columns, colors_p):
    ax4.plot(monthly.index, monthly[col]/1e3, marker='o', markersize=4,
             linewidth=1.8, label=col, color=col_color)
    ax4.fill_between(monthly.index, monthly[col]/1e3, alpha=0.08, color=col_color)
ax4.set_xticks(range(1, 13)); ax4.set_xticklabels(month_names, fontsize=8)
ax4.set_title('Monthly Revenue Trend by Product ($K)', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax4.set_ylabel('Revenue ($K)', fontsize=8)
ax4.legend(fontsize=8, loc='upper left', framealpha=0.3,
           facecolor=PANEL, edgecolor='#30363d')
ax4.grid(True)

# ── PLOT 5: Satisfaction Histogram + KDE ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df['satisfaction'], bins=20, color=ACC4, alpha=0.55,
         edgecolor='#0d1117', density=True, zorder=2)
sns.kdeplot(df['satisfaction'], ax=ax5, color=ACC5, linewidth=2, zorder=3)
mean_sat = df['satisfaction'].mean()
ax5.axvline(mean_sat, color=ACC2, linewidth=1.5, linestyle='--', label=f'Mean {mean_sat:.2f}')
ax5.set_title('Customer Satisfaction\nDistribution + KDE', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax5.set_xlabel('Rating (1–5)', fontsize=8)
ax5.set_ylabel('Density', fontsize=8)
ax5.legend(fontsize=8, framealpha=0.3, facecolor=PANEL, edgecolor='#30363d')
ax5.grid(axis='y')

# ── PLOT 6: Discount Tier × Revenue Category Stacked Bar ─────────────────────
ax6 = fig.add_subplot(gs[2, 0])
cross = pd.crosstab(df['discount_tier'], df['revenue_category'], normalize='index') * 100
cross[['Premium','Standard']].plot(kind='bar', ax=ax6, stacked=True,
      color=[ACC1, ACC3], edgecolor='#0d1117', linewidth=0.6)
ax6.set_title('Revenue Category by\nDiscount Tier (%)', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax6.set_ylabel('Proportion (%)', fontsize=8)
ax6.set_xlabel('Discount Tier', fontsize=8)
ax6.tick_params(axis='x', rotation=0)
ax6.legend(fontsize=8, framealpha=0.3, facecolor=PANEL, edgecolor='#30363d')
ax6.grid(axis='y', zorder=0)

# ── PLOT 7: Pivot Table Heatmap (Avg Revenue) ─────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
pivot_data = df.pivot_table(values='revenue', index='region',
                             columns='product', aggfunc='mean')
sns.heatmap(pivot_data/1e3, ax=ax7, cmap='YlOrRd', annot=True,
            fmt='.1f', annot_kws={'size': 8}, linewidths=0.5,
            linecolor='#0d1117', cbar_kws={'label':'Avg Rev ($K)', 'shrink':0.8})
ax7.set_title('Pivot: Avg Revenue ($K)\nRegion × Product', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax7.tick_params(axis='x', rotation=30, labelsize=7.5)
ax7.tick_params(axis='y', rotation=0, labelsize=7.5)

# ── PLOT 8: Z-score Distribution (Standardised) ──────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
z_cols   = ['units_sold_z', 'unit_price_z', 'revenue_z']
z_colors = [ACC2, ACC5, ACC1]
for col, c in zip(z_cols, z_colors):
    sns.kdeplot(scaled[col], ax=ax8, color=c, linewidth=2,
                label=col.replace('_z', '').replace('_', ' ').title(), fill=True, alpha=0.15)
ax8.axvline(0, color='#ffffff', linewidth=0.8, linestyle='--', alpha=0.5)
ax8.set_title('Standardised Features\n(Z-score KDE)', fontsize=11,
              fontweight='bold', pad=8, color='#ffffff')
ax8.set_xlabel('Z-score', fontsize=8)
ax8.set_ylabel('Density', fontsize=8)
ax8.legend(fontsize=8, framealpha=0.3, facecolor=PANEL, edgecolor='#30363d')
ax8.grid(axis='y')

plt.savefig('retail_dashboard.png',
            dpi=160, bbox_inches='tight', facecolor=DARK)
print("\nDashboard saved as 'retail_dashboard.png'")
plt.show()