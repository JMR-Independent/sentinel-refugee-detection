"""Regenerate Figure 1 (roble1) for GRSL paper.

Three panels:
(a) NDBI spectral gap vs LOCO AUC scatter (r=0.912)
(b) ROC curves for South Sudan: spectral vs texture
(c) Cohen's d for GLCM homogeneity by country (texture inversion)
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent

# =====================================================================
# DATA FOR PANEL (a): NDBI gap vs LOCO AUC
# =====================================================================
countries_a = ['Chad', 'Ethiopia', 'S. Sudan', 'Syria', 'Turkey', 'Uganda', 'Yemen']
gaps =        [+0.0455, +0.0233, -0.0770, -0.0009, +0.1288, +0.0549, +0.0201]
aucs =        [ 0.652,   0.678,   0.238,   0.531,   0.773,   0.654,   0.658]
colors_a =    ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd', '#e41a1c', '#2ca02c', '#17becf']

# =====================================================================
# DATA FOR PANEL (b): ROC curves South Sudan
# =====================================================================
# Texture ROC from saved NPZ
roc_npz = np.load(PROJECT / 'data' / 'analysis' / 'roc_data_south_sudan_texture.npz')
fpr_texture = roc_npz['fpr']
tpr_texture = roc_npz['tpr']

# Spectral ROC: compute from tiles using NDBI LogReg (LOCO for S. Sudan)
with open(PROJECT / 'data' / 'analysis' / 'texture_features.json') as f:
    all_features = json.load(f)

# Get country and label info for all tiles
tile_info = {d['tile_id']: {'country': d['country'], 'label': d['label']} for d in all_features}

tile_dir = PROJECT / 'data' / 'sentinel2'

# Compute NDBI features for all tiles
ndbi_features = []
labels_all = []
countries_all = []

for tid, info in tile_info.items():
    tile_path = tile_dir / f'{tid}.npy'
    if not tile_path.exists():
        continue
    arr = np.load(tile_path)
    ndbi_channel = arr[4]  # Channel 4 = NDBI
    ndbi_mean = ndbi_channel.mean()
    ndbi_std = ndbi_channel.std()
    ndbi_frac = (ndbi_channel > 0).mean()
    ndbi_features.append([ndbi_mean, ndbi_std, ndbi_frac])
    labels_all.append(1 if info['label'] == 'camp' else 0)
    countries_all.append(info['country'])

ndbi_features = np.array(ndbi_features)
labels_all = np.array(labels_all)
countries_all = np.array(countries_all)

# LOCO for South Sudan: train on other 6, test on S. Sudan
train_mask = countries_all != 'south_sudan'
test_mask = countries_all == 'south_sudan'

X_train, y_train = ndbi_features[train_mask], labels_all[train_mask]
X_test, y_test = ndbi_features[test_mask], labels_all[test_mask]

clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
fpr_spectral, tpr_spectral, _ = roc_curve(y_test, y_prob)
auc_spectral = roc_auc_score(y_test, y_prob)
print(f"S. Sudan spectral AUC: {auc_spectral:.3f}")

# =====================================================================
# DATA FOR PANEL (c): Cohen's d GLCM homogeneity per country
# =====================================================================
country_order_c = ['S. Sudan', 'Syria', 'Yemen', 'Ethiopia', 'Turkey', 'Uganda', 'Chad']
country_keys_c  = ['south_sudan', 'syria', 'yemen', 'ethiopia', 'turkey', 'uganda', 'chad']

cohens_d = []
for ck in country_keys_c:
    camp_vals = np.array([d['glcm_homogeneity'] for d in all_features
                          if d['country'] == ck and d['label'] == 'camp'])
    neg_vals = np.array([d['glcm_homogeneity'] for d in all_features
                         if d['country'] == ck and d['label'] != 'camp'])
    pooled_std = np.sqrt((camp_vals.std()**2 * (len(camp_vals)-1) +
                          neg_vals.std()**2 * (len(neg_vals)-1)) /
                         (len(camp_vals) + len(neg_vals) - 2))
    d_val = (camp_vals.mean() - neg_vals.mean()) / pooled_std if pooled_std > 0 else 0
    cohens_d.append(d_val)
    print(f"  {ck:15s}  d={d_val:+.3f}")

# =====================================================================
# FIGURE
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8),
                         gridspec_kw={'width_ratios': [1.2, 1, 1]})

# --- Panel (a): Scatter ---
ax = axes[0]
for i, (g, a, c, name) in enumerate(zip(gaps, aucs, colors_a, countries_a)):
    ax.scatter(g, a, c=c, s=70, zorder=5, edgecolors='k', linewidth=0.5)
    # Offset labels to avoid overlap in the dense cluster
    offset_x, offset_y = 0.004, 0.018
    if name == 'Syria':
        offset_x, offset_y = 0.005, -0.035
    elif name == 'S. Sudan':
        offset_x, offset_y = 0.010, 0.020
    elif name == 'Chad':
        offset_x, offset_y = 0.005, -0.030
    elif name == 'Yemen':
        offset_x, offset_y = -0.028, 0.020
    elif name == 'Ethiopia':
        offset_x, offset_y = 0.005, 0.020
    elif name == 'Uganda':
        offset_x, offset_y = 0.005, -0.005
    elif name == 'Turkey':
        offset_x, offset_y = -0.020, 0.020
    ax.annotate(name, (g, a), xytext=(g + offset_x, a + offset_y),
                fontsize=7.5, fontstyle='italic')

# Regression line
slope, intercept, r, p, se = stats.linregress(gaps, aucs)
x_line = np.linspace(-0.10, 0.15, 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.4, linewidth=1)

# Diagonal reference
ax.plot([-0.10, 0.15], [0.5, 0.5], ':', color='gray', alpha=0.3, linewidth=0.8)

ax.set_xlabel(r'$\Delta$NDBI (camp $-$ background)', fontsize=9)
ax.set_ylabel('LOCO AUC', fontsize=9)
ax.set_xlim(-0.10, 0.15)
ax.set_ylim(0.15, 0.85)

# Stats box
stats_text = f'r = {r:.3f}\nR² = {r**2:.3f}\np = 0.005'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
        facecolor='white', edgecolor='gray', alpha=0.8))
ax.text(0.02, 1.03, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')

# --- Panel (b): ROC South Sudan ---
ax = axes[1]
ax.plot(fpr_spectral, tpr_spectral, 'b-', linewidth=1.5,
        label=f'Spectral ({auc_spectral:.2f})')
ax.plot(fpr_texture, tpr_texture, 'r--', linewidth=1.5,
        label=f'Texture ({roc_npz["auc"]:.2f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
ax.set_xlabel('False Positive Rate', fontsize=9)
ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title('(b) South Sudan', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# --- Panel (c): Cohen's d bars ---
ax = axes[2]
bar_colors = []
for d_val in cohens_d:
    bar_colors.append('#2196F3' if d_val > 0 else '#FF9800')

y_pos = np.arange(len(country_order_c))
ax.barh(y_pos, cohens_d, color=bar_colors, edgecolor='k', linewidth=0.5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(country_order_c, fontsize=8)
ax.set_xlabel("Cohen's d (GLCM homogeneity)", fontsize=9)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-0.8, 0.9)
ax.text(0.02, 1.03, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()

# Save in multiple formats
for ext in ['png', 'pdf', 'tiff']:
    out = PROJECT / 'paper' / f'roble1.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')

plt.close()
print('Done.')
