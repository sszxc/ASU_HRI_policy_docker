// Polls /ood/status the same way app.js polls /api/status -- no websocket, matches
// the rest of this package's web pages.
const POLL_MS = 300;
const TITLE_OVERRIDES = { qpos: "qpos" }; // everything else is a camera name, used as-is

let modalities = [];

function cloudTrace(points) {
  return {
    x: points.map((p) => p[0]),
    y: points.map((p) => p[1]),
    mode: "markers",
    type: "scattergl",
    marker: { size: 4, color: "#3b6", opacity: 0.35 },
    name: "training",
    hoverinfo: "skip",
  };
}

function liveTrace() {
  return {
    x: [],
    y: [],
    mode: "markers",
    type: "scatter",
    marker: { size: 14, color: "#888", line: { color: "#fff", width: 1 } },
    name: "live",
    hoverinfo: "skip",
  };
}

async function init() {
  const ref = await (await fetch("/ood/reference")).json();
  modalities = ref.modalities;
  const panels = document.getElementById("panels");
  modalities.forEach((name) => {
    const div = document.createElement("div");
    div.className = "panel";
    div.id = `plot-${name}`;
    panels.append(div);
    Plotly.newPlot(
      div.id,
      [cloudTrace(ref.background[name]), liveTrace()],
      {
        title: TITLE_OVERRIDES[name] || `cam: ${name}`,
        margin: { t: 32, l: 32, r: 8, b: 32 },
        paper_bgcolor: "#111",
        plot_bgcolor: "#1a1a1a",
        font: { color: "#eee" },
        showlegend: false,
      },
      { displayModeBar: false, responsive: true }
    );
  });
  poll();
}

// Rough visual cue only -- ratio to the fixed SCALE constant below is not a
// calibrated threshold. Read the raw numbers (and the logged history) before
// trusting a color; recalibrate SCALE once real-run distributions are in.
const SCALE = 1.0;
function colorFor(dist) {
  if (dist == null) return "#888";
  const ratio = dist / SCALE;
  if (ratio < 1.5) return "#3b6";
  if (ratio < 3) return "#e2a03f";
  return "#e24";
}

async function poll() {
  try {
    const status = await (await fetch("/ood/status")).json();
    document.getElementById("scores").innerHTML = modalities
      .map((name) => {
        const d = status.knn_dist[name];
        return `<div><span class="label">${TITLE_OVERRIDES[name] || name}</span>${d == null ? "--" : d.toFixed(3)}</div>`;
      })
      .join("");
    modalities.forEach((name) => {
      const xy = status.embed[name];
      if (!xy) return;
      Plotly.restyle(`plot-${name}`, { x: [[xy[0]]], y: [[xy[1]]], "marker.color": [colorFor(status.knn_dist[name])] }, [1]);
    });
  } catch (err) {
    // transient fetch error (server restarting, etc.) -- just retry next tick
  }
  setTimeout(poll, POLL_MS);
}

init();
