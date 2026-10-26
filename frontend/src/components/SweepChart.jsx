import React from 'react';
import Plot from 'react-plotly.js';

export default function SweepChart({ sweepData }) {
  if (!sweepData || sweepData.length === 0) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Run parameter sweep to view Elbow & Silhouette curves.</div>;
  }

  const kValues = sweepData.map((d) => d.k);
  const inertiaValues = sweepData.map((d) => d.inertia);
  const silhouetteValues = sweepData.map((d) => d.silhouette);

  const traces = [
    {
      x: kValues,
      y: inertiaValues,
      name: 'Inertia (WCSS - Elbow)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#18181b', width: 2.5 },
      marker: { size: 8, color: '#18181b' },
      yaxis: 'y1',
    },
    {
      x: kValues,
      y: silhouetteValues,
      name: 'Cosine Silhouette',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#71717a', width: 2, dash: 'dot' },
      marker: { size: 8, color: '#71717a' },
      yaxis: 'y2',
    },
  ];

  const layout = {
    autosize: true,
    height: 320,
    margin: { l: 50, r: 50, b: 40, t: 30 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {
      title: 'Cluster Count (K)',
      tickmode: 'linear',
      dtick: 1,
      gridcolor: '#e4e4e7',
    },
    yaxis: {
      title: 'Inertia (WCSS)',
      titlefont: { color: '#18181b' },
      tickfont: { color: '#18181b' },
      gridcolor: '#e4e4e7',
    },
    yaxis2: {
      title: 'Silhouette Score',
      titlefont: { color: '#71717a' },
      tickfont: { color: '#71717a' },
      overlaying: 'y',
      side: 'right',
      range: [-0.1, 0.4],
      showgrid: false,
    },
    legend: {
      orientation: 'h',
      y: 1.15,
      x: 0,
    },
  };

  return (
    <div style={{ width: '100%', height: '320px' }}>
      <Plot
        data={traces}
        layout={layout}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
