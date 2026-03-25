import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Ladda data från prices.csv
df = pd.read_csv('prices.csv', index_col='Date', parse_dates=True)

# 2. Identifiera alla tillgångskolumner (Stocks, FX, Comm, Idx)
asset_cols = [c for c in df.columns if any(c.startswith(p) for p in ['Stock_', 'FX_', 'Comm_', 'Idx_'])]

# 3. Beräkna daglig avkastning och korrelation
returns = df[asset_cols].pct_change().dropna()
corr_matrix = returns.corr()

# 4. Skapa plotten
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

# Lägg till en färgskala
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

# Fixa axlarna och etiketter
ax.set_xticks(np.arange(len(asset_cols)))
ax.set_yticks(np.arange(len(asset_cols)))
ax.set_xticklabels(asset_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(asset_cols, fontsize=8)

ax.set_title("Asset Correlation Matrix (Daily Returns)")
fig.tight_layout()

# Spara bilden
plt.savefig('asset_correlation.png')