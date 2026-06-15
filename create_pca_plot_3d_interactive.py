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

legend_symbols = {
    1: '●',
    3: '■',
    2: '✚',
    4: '◆',
}

MARKER_SIZE = 6.5
LEGEND_SYMBOL_SIZE = 24
LEGEND_FONT_SIZE = 17

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
            showlegend=False,
            marker=dict(
                size=MARKER_SIZE,
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

# Custom legend: colored symbols + cluster names (top-left)
legend_annotations = []
legend_y_start = 0.975
legend_row_step = 0.058
n_rows = len(cluster_order)

for i, cluster in enumerate(cluster_order):
    color = colors_new[cluster]
    name = cluster_names[cluster]
    y = legend_y_start - i * legend_row_step

    legend_annotations.append(dict(
        xref='paper', yref='paper',
        x=0.028, y=y,
        text=legend_symbols[cluster],
        font=dict(size=LEGEND_SYMBOL_SIZE, color=color, family='Arial Black, Arial, sans-serif'),
        showarrow=False,
        xanchor='center',
        yanchor='middle',
    ))
    legend_annotations.append(dict(
        xref='paper', yref='paper',
        x=0.052, y=y,
        text=f'<b>{name}</b>',
        font=dict(size=LEGEND_FONT_SIZE, color=color, family='Arial Black, Arial, sans-serif'),
        showarrow=False,
        xanchor='left',
        yanchor='middle',
    ))

legend_shapes = [dict(
    type='rect',
    xref='paper', yref='paper',
    x0=0.008, y0=legend_y_start - (n_rows - 1) * legend_row_step - 0.028,
    x1=0.155, y1=legend_y_start + 0.028,
    line=dict(color='gray', width=1),
    fillcolor='rgba(255,255,255,0.92)',
    layer='below',
)]

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
    showlegend=False,
    annotations=legend_annotations,
    shapes=legend_shapes,
    margin=dict(l=0, r=0, t=60, b=0),
    paper_bgcolor='white',
    width=1100,
    height=800,
)

output_file = 'Cluster_Separation_PCA_3D_interactive.html'
fig.write_html(output_file, include_plotlyjs=True, full_html=True)
print(f"Interactive 3D PCA plot saved as '{output_file}'")
print("Open in a browser: drag to rotate, scroll to zoom.")
