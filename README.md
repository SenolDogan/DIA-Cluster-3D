# DIA Cluster 3D — Interactive PCA Visualization

Interactive three-dimensional principal component analysis (PCA) plot showing cluster separation in diabetes subtypes (SIDD, SIRD, MOD, MARD).

## Interactive figure (for manuscript readers)

**Live link:** https://senoldogan.github.io/DIA-Cluster-3D/

Open in any web browser. Drag to rotate the 3D plot; scroll to zoom; hover over points for cluster details.

## Files

| File | Description |
|------|-------------|
| `index.html` | Interactive 3D PCA figure (main entry point for GitHub Pages) |
| `Cluster_Separation_PCA_3D_interactive.html` | Same interactive figure |
| `create_pca_plot_3d_interactive.py` | Python script to regenerate the figure |

## Regenerate locally

```bash
pip install plotly pandas scikit-learn
python create_pca_plot_3d_interactive.py
```

Requires `Dataset.csv` in the same directory (not included in this repository).

## Citation (manuscript supplementary material)

> **Supplementary Material S1.** Interactive three-dimensional PCA plot of cluster separation. Available at: https://senoldogan.github.io/DIA-Cluster-3D/ (accessed [date]).

## License

For use as supplementary material accompanying the associated manuscript.
