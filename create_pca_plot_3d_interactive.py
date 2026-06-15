"""
Interactive 3D PCA cluster plot.
Drag to rotate, scroll to zoom — data points move with the view.
Output: Cluster_Separation_PCA_3D_interactive.html
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Read the dataset
df = pd.read_csv('Dataset.csv')

features = ['age', 'mean_BMI', 'Trig/HDL ratio', 'HbA1c']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

cluster_names = {
    1: 'SIDD',
    2: 'MOD',
    3: 'SIRD',
    4: 'MARD',
}

colors_new = {
    1: '#A6CEE3',  # SIDD
    2: '#FBB4AE',  # MOD
    3: '#B2DF8A',  # SIRD
    4: '#CAB2D6',  # MARD
}

cluster_order = [1, 3, 2, 4]  # SIDD, SIRD, MOD, MARD

plotly_symbols = {
    1: 'circle',
    3: 'square',
    2: 'cross',      # triangle not available in Plotly 3D
    4: 'diamond',
}

fig = go.Figure()

for cluster in cluster_order:
    mask = df['cluster'] == cluster
    fig.add_trace(
        go.Scatter3d(
            x=X_pca_3d[mask, 0],
            y=X_pca_3d[mask, 1],
            z=X_pca_3d[mask, 2],
            mode='markers',
            name=cluster_names[cluster],
            marker=dict(
                size=4,
                color=colors_new[cluster],
                symbol=plotly_symbols[cluster],
                line=dict(color='black', width=0.5),
                opacity=0.75,
            ),
            hovertemplate=(
                f'<b>{cluster_names[cluster]}</b><br>'
                'PC1: %{x:.2f}<br>'
                'PC2: %{y:.2f}<br>'
                'PC3: %{z:.2f}<extra></extra>'
            ),
        )
    )

# Cluster centroids
for cluster in cluster_order:
    mask = df['cluster'] == cluster
    fig.add_trace(
        go.Scatter3d(
            x=[X_pca_3d[mask, 0].mean()],
            y=[X_pca_3d[mask, 1].mean()],
            z=[X_pca_3d[mask, 2].mean()],
            mode='markers',
            name=f'{cluster_names[cluster]} centroid',
            marker=dict(size=6, color='black', symbol='x', line=dict(width=2)),
            showlegend=False,
            hoverinfo='skip',
        )
    )

var = pca_3d.explained_variance_ratio_
fig.update_layout(
    title=dict(
        text='Cluster Separation - PCA Analysis (3D)',
        font=dict(size=18, family='Arial Black, Arial, sans-serif'),
        x=0.5,
    ),
    scene=dict(
        xaxis=dict(
            title=dict(
                text=f'PC1 ({var[0]:.2%} variance)',
                font=dict(size=14, family='Arial Black, Arial, sans-serif'),
            ),
            backgroundcolor='white',
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title=dict(
                text=f'PC2 ({var[1]:.2%} variance)',
                font=dict(size=14, family='Arial Black, Arial, sans-serif'),
            ),
            backgroundcolor='white',
            gridcolor='lightgray',
        ),
        zaxis=dict(
            title=dict(
                text=f'PC3 ({var[2]:.2%} variance)',
                font=dict(size=14, family='Arial Black, Arial, sans-serif'),
            ),
            backgroundcolor='white',
            gridcolor='lightgray',
        ),
        bgcolor='white',
    ),
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1,
        font=dict(size=14, family='Arial Black, Arial, sans-serif'),
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    paper_bgcolor='white',
    width=1100,
    height=800,
)

output_file = 'Cluster_Separation_PCA_3D_interactive.html'
fig.write_html(output_file, include_plotlyjs=True, full_html=True)
print(f"Interactive 3D PCA plot saved as '{output_file}'")
print("Open in a browser: drag to rotate, scroll to zoom.")
