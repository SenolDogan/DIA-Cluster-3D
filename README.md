# DIA Cluster 3D — Interactive PCA Visualization

Interactive three-dimensional principal component analysis (PCA) plot showing cluster separation in diabetes subtypes (SIDD, SIRD, MOD, MARD).

## Interactive figure (click to open in browser)

**Manuscript link:** https://senoldogan.github.io/DIA-Cluster-3D/

> If the link above returns 404, enable GitHub Pages once:  
> Repository **Settings → Pages → Build and deployment → Deploy from branch → `gh-pages` / `/ (root)` → Save**

**Alternative (works immediately):**  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/SenolDogan/DIA-Cluster-3D/main/index.html

Drag to rotate · scroll to zoom · hover for cluster details.

## Repository

https://github.com/SenolDogan/DIA-Cluster-3D

## Manuscript citation

> **Supplementary Material S1.** Interactive three-dimensional PCA plot of cluster separation. Available at: https://senoldogan.github.io/DIA-Cluster-3D/

See `MANUSCRIPT_CITATION.txt` for ready-to-paste text (Vancouver, APA, figure legend).

## Regenerate locally

```bash
pip install plotly pandas scikit-learn
python create_pca_plot_3d_interactive.py
```

Requires `Dataset.csv` in the same directory (not included in this repository).
