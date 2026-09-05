// Polls /ood/status the way app.js polls /api/status -- no websocket, and the scatter
// is drawn on a plain canvas so the page needs no CDN/internet on the viewing machine.
const POLL_MS = 200;
const TRAIL_LEN = 120;

const panels = {};   // modality -> {canvas, ctx, cloud (offscreen), bounds, trail}
let modalities = [];
let baseline = {};

function bounds(points) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const padX = (x1 - x0) * 0.08 || 1;
  const padY = (y1 - y0) * 0.08 || 1;
  return { x0: x0 - padX, x1: x1 + padX, y0: y0 - padY, y1: y1 + padY };
}

function project(panel, xy) {
  const { x0, x1, y0, y1 } = panel.bounds;
  const w = panel.canvas.width / panel.dpr;
  const h = panel.canvas.height / panel.dpr;
  return [((xy[0] - x0) / (x1 - x0)) * w, h - ((xy[1] - y0) / (y1 - y0)) * h];
}

// Training cloud never changes -- render it once into an offscreen canvas and blit.
function renderCloud(panel, points) {
  const cloud = document.createElement("canvas");
  cloud.width = panel.canvas.width;
  cloud.height = panel.canvas.height;
  const ctx = cloud.getContext("2d");
  ctx.scale(panel.dpr, panel.dpr);
  ctx.fillStyle = "rgba(150, 165, 175, 0.28)";
  points.forEach((p) => {
    const [x, y] = project(panel, p);
    ctx.fillRect(x - 1, y - 1, 2.5, 2.5);
  });
  panel.cloud = cloud;
}

function resize(panel, points) {
  const dpr = window.devicePixelRatio || 1;
  const rect = panel.canvas.getBoundingClientRect();
  panel.dpr = dpr;
  panel.canvas.width = Math.max(1, Math.round(rect.width * dpr));
  panel.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  panel.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  renderCloud(panel, points);
  draw(panel);
}

function colorFor(name, dist) {
  if (dist == null || !baseline[name]) return "#9ba6ad";
  const ratio = dist / baseline[name]["95"];
  if (ratio < 1.0) return "#41c27d";
  if (ratio < 2.0) return "#e4ad49";
  return "#ef5a5a";
}

function draw(panel) {
  const ctx = panel.ctx;
  const w = panel.canvas.width / panel.dpr;
  const h = panel.canvas.height / panel.dpr;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(panel.cloud, 0, 0);
  ctx.restore();
  if (!panel.trail.length) return;

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(86, 185, 208, 0.55)";
  ctx.beginPath();
  panel.trail.forEach((p, i) => {
    const [x, y] = project(panel, p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const [x, y] = project(panel, panel.trail[panel.trail.length - 1]);
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fillStyle = panel.color;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#f2f4f5";
  ctx.stroke();
}

async function init() {
  const ref = await (await fetch("/ood/reference")).json();
  modalities = ref.modalities;
  baseline = ref.baseline;

  const scoresEl = document.getElementById("scores");
  const panelsEl = document.getElementById("panels");
  modalities.forEach((name) => {
    scoresEl.insertAdjacentHTML("beforeend",
      `<div class="score"><div class="name">${name === "qpos" ? "qpos" : `cam: ${name}`}</div>` +
      `<div class="dist" id="dist-${name}">--</div><div class="ratio" id="ratio-${name}">&nbsp;</div></div>`);

    const div = document.createElement("div");
    div.className = "panel";
    div.innerHTML = `<h2>${name}</h2><canvas></canvas>` +
      `<div class="legend">grey = training set &middot; cyan = recent trail &middot; dot = now</div>`;
    panelsEl.append(div);

    const canvas = div.querySelector("canvas");
    const panel = { canvas, ctx: canvas.getContext("2d"), dpr: 1, trail: [], color: "#9ba6ad" };
    panel.bounds = bounds(ref.background[name]);
    panels[name] = panel;
    resize(panel, ref.background[name]);
    window.addEventListener("resize", () => resize(panel, ref.background[name]));
  });
  poll();
}

async function poll() {
  try {
    const status = await (await fetch("/ood/status")).json();
    document.getElementById("conn").textContent = status.t ? "live" : "waiting for data...";
    modalities.forEach((name) => {
      const dist = status.knn_dist[name];
      const panel = panels[name];
      document.getElementById(`dist-${name}`).textContent = dist == null ? "--" : dist.toFixed(3);
      document.getElementById(`ratio-${name}`).textContent =
        dist == null || !baseline[name] ? " " : `${(dist / baseline[name]["95"]).toFixed(2)}x train p95`;
      panel.color = colorFor(name, dist);
      const xy = status.embed[name];
      const last = panel.trail[panel.trail.length - 1];
      if (xy && (!last || last[0] !== xy[0] || last[1] !== xy[1])) {
        panel.trail.push(xy);
        if (panel.trail.length > TRAIL_LEN) panel.trail.shift();
      }
      draw(panel);
    });
  } catch (err) {
    document.getElementById("conn").textContent = "disconnected";
  }
  setTimeout(poll, POLL_MS);
}

init();
