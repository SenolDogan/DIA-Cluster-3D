"""
Interactive 3D PCA cluster plot.
Drag to rotate, scroll to zoom — data points move with the view.
Click legend items to show/hide clusters.
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

MARKER_SIZE = 6.5
LEGEND_FONT_SIZE = 18

fig = go.Figure()

for cluster in cluster_order:
    mask = df['cluster'] == cluster
    group = cluster_names[cluster]

    fig.add_trace(
        go.Scatter3d(
            x=X_pca_3d[mask, 0],
            y=X_pca_3d[mask, 1],
            z=X_pca_3d[mask, 2],
            mode='markers',
            name=group,
            legendgroup=group,
            showlegend=True,
            marker=dict(
                size=MARKER_SIZE,
                color=colors_new[cluster],
                symbol=plotly_symbols[cluster],
                line=dict(color='black', width=0.5),
                opacity=0.75,
            ),
            hovertemplate=(
                f'<b>{group}</b><br>'
                'PC1: %{x:.2f}<br>'
                'PC2: %{y:.2f}<br>'
                'PC3: %{z:.2f}<extra></extra>'
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[X_pca_3d[mask, 0].mean()],
            y=[X_pca_3d[mask, 1].mean()],
            z=[X_pca_3d[mask, 2].mean()],
            mode='markers',
            name=f'{group} centroid',
            legendgroup=group,
            showlegend=False,
            marker=dict(size=6, color='black', symbol='x', line=dict(width=2)),
            hoverinfo='skip',
        )
    )

var = pca_3d.explained_variance_ratio_
fig.update_layout(
    title=dict(
        text='Cluster Separation - PCA Analysis (3D)',
        font=dict(size=18, family='Arial Black, Arial, sans-serif'),
        x=0.5,
        y=0.97,
    ),
    scene=dict(
        domain=dict(x=[0.0, 1.0], y=[0.06, 0.96]),
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
        x=0.01,
        y=0.99,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.92)',
        bordercolor='gray',
        borderwidth=1,
        font=dict(size=LEGEND_FONT_SIZE, family='Arial Black, Arial, sans-serif'),
        itemsizing='constant',
        itemwidth=40,
        tracegroupgap=8,
        itemclick='toggle',
        itemdoubleclick='toggleothers',
    ),
    margin=dict(l=0, r=0, t=40, b=20),
    paper_bgcolor='white',
    width=1100,
    height=800,
)

output_file = 'Cluster_Separation_PCA_3D_interactive.html'
fig.write_html(output_file, include_plotlyjs=True, full_html=True)
print(f"Interactive 3D PCA plot saved as '{output_file}'")
print("Open in a browser: drag to rotate, scroll to zoom, click legend to toggle clusters.")
