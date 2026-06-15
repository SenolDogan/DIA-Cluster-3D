"""
Interactive 3D PCA cluster plot.
Drag to rotate, scroll to zoom. Click custom legend to show/hide clusters.
Output: Cluster_Separation_PCA_3D_interactive.html
"""

import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

df = pd.read_csv('Dataset.csv')

features = ['age', 'mean_BMI', 'Trig/HDL ratio', 'HbA1c']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

cluster_names = {1: 'SIDD', 2: 'MOD', 3: 'SIRD', 4: 'MARD'}
colors_new = {1: '#A6CEE3', 2: '#FBB4AE', 3: '#B2DF8A', 4: '#CAB2D6'}
cluster_order = [1, 3, 2, 4]

plotly_symbols = {1: 'circle', 3: 'square', 2: 'cross', 4: 'diamond'}
legend_symbols = {1: '●', 3: '■', 2: '✚', 4: '◆'}

MARKER_SIZE = 6.5
LEGEND_FONT_SIZE = 18
LEGEND_SYMBOL_SIZE = 22

fig = go.Figure()
trace_map = {}

for cluster in cluster_order:
    mask = df['cluster'] == cluster
    group = cluster_names[cluster]
    trace_indices = []

    fig.add_trace(
        go.Scatter3d(
            x=X_pca_3d[mask, 0],
            y=X_pca_3d[mask, 1],
            z=X_pca_3d[mask, 2],
            mode='markers',
            name=group,
            showlegend=False,
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
    trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter3d(
            x=[X_pca_3d[mask, 0].mean()],
            y=[X_pca_3d[mask, 1].mean()],
            z=[X_pca_3d[mask, 2].mean()],
            mode='markers',
            name=f'{group} centroid',
            showlegend=False,
            marker=dict(size=6, color='black', symbol='x', line=dict(width=2)),
            hoverinfo='skip',
        )
    )
    trace_indices.append(len(fig.data) - 1)
    trace_map[group] = trace_indices

var = pca_3d.explained_variance_ratio_
fig.update_layout(
    title=dict(
        text='Cluster Separation - PCA Analysis (3D)',
        font=dict(size=18, family='Arial Black, Arial, sans-serif'),
        x=0.5,
        y=0.98,
        pad=dict(t=2, b=0),
    ),
    scene=dict(
        domain=dict(x=[0.0, 1.0], y=[0.0, 0.93]),
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
    margin=dict(l=0, r=0, t=22, b=0),
    paper_bgcolor='white',
    width=1100,
    height=800,
)

plot_div_id = 'pca-plot-3d'
plot_html = fig.to_html(
    include_plotlyjs=True,
    full_html=False,
    div_id=plot_div_id,
    config={'scrollZoom': True, 'displayModeBar': True},
)

legend_items_html = []
for cluster in cluster_order:
    group = cluster_names[cluster]
    color = colors_new[cluster]
    sym = legend_symbols[cluster]
    legend_items_html.append(
        f'<div class="legend-item active" data-cluster="{group}" '
        f'style="color:{color};" title="Click to show/hide">'
        f'<span class="legend-symbol" style="color:{color};font-size:{LEGEND_SYMBOL_SIZE}px;">{sym}</span>'
        f'<span class="legend-label" style="color:{color};font-size:{LEGEND_FONT_SIZE}px;">{group}</span>'
        f'</div>'
    )

trace_map_json = json.dumps(trace_map)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Cluster Separation - PCA Analysis (3D)</title>
<style>
  body {{
    margin: 0;
    background: #fff;
    font-family: Arial, sans-serif;
  }}
  #figure-wrapper {{
    position: relative;
    width: 1100px;
    height: 800px;
    margin: 0 auto;
  }}
  #custom-legend {{
    position: absolute;
    top: 52px;
    left: 12px;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #888;
    border-radius: 4px;
    padding: 10px 14px 8px 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    pointer-events: auto;
    user-select: none;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    cursor: pointer;
    font-weight: bold;
    line-height: 1.1;
  }}
  .legend-item.inactive {{
    opacity: 0.35;
  }}
  .legend-item.inactive .legend-label,
  .legend-item.inactive .legend-symbol {{
    text-decoration: line-through;
  }}
  .legend-symbol {{
    width: 26px;
    text-align: center;
    display: inline-block;
  }}
  #pca-plot-3d {{
    width: 100%;
    height: 100%;
  }}
</style>
</head>
<body>
<div id="figure-wrapper">
  <div id="custom-legend">
    {''.join(legend_items_html)}
  </div>
  {plot_html}
</div>
<script>
(function() {{
  const PLOT_ID = '{plot_div_id}';
  const TRACE_MAP = {trace_map_json};
  const visibility = {{}};
  Object.keys(TRACE_MAP).forEach(function(name) {{ visibility[name] = true; }});

  function setClusterVisible(name, visible) {{
    const indices = TRACE_MAP[name];
    Plotly.restyle(PLOT_ID, {{visible: visible}}, indices);
    visibility[name] = visible;
    const el = document.querySelector('.legend-item[data-cluster="' + name + '"]');
    if (el) {{
      el.classList.toggle('active', visible);
      el.classList.toggle('inactive', !visible);
    }}
  }}

  document.querySelectorAll('.legend-item').forEach(function(item) {{
    item.addEventListener('click', function(e) {{
      e.preventDefault();
      e.stopPropagation();
      const name = item.getAttribute('data-cluster');
      setClusterVisible(name, !visibility[name]);
    }});
    item.addEventListener('dblclick', function(e) {{
      e.preventDefault();
      e.stopPropagation();
      const name = item.getAttribute('data-cluster');
      Object.keys(TRACE_MAP).forEach(function(other) {{
        setClusterVisible(other, other === name);
      }});
    }});
  }});
}})();
</script>
</body>
</html>
"""

output_file = 'Cluster_Separation_PCA_3D_interactive.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Interactive 3D PCA plot saved as '{output_file}'")
print("Custom legend: click to toggle, double-click to isolate cluster.")
