const GRID_BY_GROUP = { external: "#externalGrid", wrist: "#wristGrid", right: "#rightGrid", tactile: "#tactileGrid" };
let cameraNames = [];
let jointNames = [];
let latestStatus = null;
const $ = (selector) => document.querySelector(selector);

function cameraElement(name) {
  const element = document.createElement("div");
  element.className = "camera";
  element.id = `camera-${name}`;
  element.innerHTML = `
    <img alt="${name} camera live view">
    <div class="camera-overlay">Waiting for stream</div>
    <div class="camera-footer">
      <span class="camera-name">${name}</span>
      <span class="stream-badge"><span class="status-pin"></span><span class="status-text">Offline</span></span>
    </div>`;
  return element;
}

function jointElement(name) {
  const row = document.createElement("div");
  row.className = "stream-row";
  row.id = `joint-${name}`;
  row.innerHTML = `
    <span><span class="status-pin"></span>${name.replace(/_/g, " ")}</span>
    <span class="joint-stream-details">
      <span class="stream-topic"></span>
      <span class="joint-diagnostics"></span>
    </span>`;
  return row;
}

async function loadConfig() {
  const config = await (await fetch("/api/config")).json();
  cameraNames = Object.keys(config.cameras);
  jointNames = Object.keys(config.joint_states);
  cameraNames.forEach((name) => {
    const grid = $(GRID_BY_GROUP[config.cameras[name].group]);
    if (grid) grid.append(cameraElement(name));
  });
  jointNames.forEach((name) => $("#jointStreams").append(jointElement(name)));
}

function formatAge(ageMs) {
  if (ageMs == null) return "never seen";
  if (ageMs < 1000) return `${Math.round(ageMs)} ms ago`;
  return `${(ageMs / 1000).toFixed(1)} s ago`;
}

function render(status) {
  latestStatus = status;
  let online = 0;
  cameraNames.forEach((name) => {
    const stream = status.streams[name];
    const element = $(`#camera-${name}`);
    const isOnline = Boolean(stream?.online);
    element.classList.toggle("online", isOnline);
    element.querySelector(".status-text").textContent = isOnline ? "Live" : "Offline";
    if (isOnline) online += 1;
  });
  $("#onlineSummary").textContent = `${online} / ${cameraNames.length} online`;
  $("#lastUpdated").textContent = new Date().toLocaleTimeString();
  jointNames.forEach((name) => {
    const row = $(`#joint-${name}`);
    const stream = status.streams[name];
    row.classList.toggle("online", Boolean(stream?.online));
    row.querySelector(".stream-topic").textContent = stream?.topic || "";
    const diagnosticsElement = row.querySelector(".joint-diagnostics");
    diagnosticsElement.textContent = stream?.online
      ? `${stream.joint_count ?? "?"} joints · ${stream.hz ?? "?"} Hz · ${formatAge(stream.age_ms)}`
      : stream?.error
        ? `error: ${stream.error}`
        : formatAge(stream?.age_ms);
    diagnosticsElement.classList.toggle("has-drops", Boolean(stream?.error));
  });
}

async function refresh() {
  try {
    render(await (await fetch("/api/status", { cache: "no-store" })).json());
  } catch (error) {
    $("#onlineSummary").textContent = `monitor unavailable: ${error.message}`;
  }
}

async function refreshCamera(name) {
  try {
    const response = await fetch(`/snapshot/${name}.jpg`, { cache: "no-store" });
    if (!response.ok) return;
    const image = $(`#camera-${name} img`);
    const previous = image.dataset.objectUrl;
    const current = URL.createObjectURL(await response.blob());
    image.src = current;
    image.dataset.objectUrl = current;
    if (previous) URL.revokeObjectURL(previous);
  } catch (_) {
    // Status polling reports connectivity; snapshot failures are transient.
  }
}

function startCameraRefresh(name) {
  const refreshLoop = async () => {
    await refreshCamera(name);
    window.setTimeout(refreshLoop, 150);
  };
  refreshLoop();
}

(async function init() {
  await loadConfig();
  cameraNames.forEach(startCameraRefresh);
  refresh();
  setInterval(refresh, 1000);
})();
