import React from 'react';
import Plot from 'react-plotly.js';

export default function PCAPlot({ points, k }) {
  if (!points || points.length === 0) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No projection points available.</div>;
  }

  // Group points by cluster_id
  const clusters = {};
  points.forEach((pt) => {
    if (!clusters[pt.cluster_id]) {
      clusters[pt.cluster_id] = {
        x: [],
        y: [],
        text: [],
        cluster_id: pt.cluster_id,
      };
    }
    clusters[pt.cluster_id].x.push(pt.x);
    clusters[pt.cluster_id].y.push(pt.y);
    clusters[pt.cluster_id].text.push(
      `<b>${pt.title}</b><br>Cluster: #${pt.cluster_id}<br>Coords: (${pt.x}, ${pt.y})<br><i>${pt.preview}</i>`
    );
  });

  // Accessible non-blue palette with distinct symbols
  const colors = ['#18181b', '#b45309', '#047857', '#be123c', '#4d7c0f', '#854d0e'];
  const symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star', 'cross'];

  const data = Object.values(clusters).map((c, i) => ({
    x: c.x,
    y: c.y,
    text: c.text,
    hoverinfo: 'text',
    mode: 'markers',
    type: 'scatter',
    name: `Cluster ${c.cluster_id}`,
    marker: {
      size: 11,
      color: colors[i % colors.length],
      symbol: symbols[i % symbols.length],
      line: {
        color: '#ffffff',
        width: 1.5,
      },
    },
  }));

  const layout = {
    autosize: true,
    height: 420,
    margin: { l: 40, r: 20, b: 40, t: 30 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    hovermode: 'closest',
    xaxis: {
      title: 'Principal Component 1 (PCA-1)',
      showgrid: true,
      gridcolor: '#e4e4e7',
      zerolinecolor: '#d1d1d6',
      color: '#71717a',
    },
    yaxis: {
      title: 'Principal Component 2 (PCA-2)',
      showgrid: true,
      gridcolor: '#e4e4e7',
      zerolinecolor: '#d1d1d6',
      color: '#71717a',
    },
    legend: {
      orientation: 'h',
      y: 1.15,
      x: 0,
    },
  };

  return (
    <div style={{ width: '100%', height: '420px' }}>
      <Plot
        data={data}
        layout={layout}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
