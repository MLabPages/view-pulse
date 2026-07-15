const MEDIAPIPE_MODULE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ANALYSIS_INTERVAL_MS = 200;
const CALIBRATION_POINTS = [
  { x: 0.18, y: 0.2 }, { x: 0.82, y: 0.2 }, { x: 0.5, y: 0.5 },
  { x: 0.18, y: 0.75 }, { x: 0.82, y: 0.75 },
];

const $ = (id) => document.getElementById(id);
const els = {
  setupScreen: $("setupScreen"), cameraScreen: $("cameraScreen"), resultsScreen: $("resultsScreen"),
  consentAnalysis: $("consentAnalysis"), saveReactionVideo: $("saveReactionVideo"),
  prepareButton: $("prepareButton"), setupStatus: $("setupStatus"),
  rearPreview: $("rearPreview"), frontPreview: $("frontPreview"), cameraModeBadge: $("cameraModeBadge"),
  closeCameraButton: $("closeCameraButton"), cameraHint: $("cameraHint"),
  calibrationLayer: $("calibrationLayer"), calibrationTarget: $("calibrationTarget"),
  calibrationProgress: $("calibrationProgress"), calibrateButton: $("calibrateButton"),
  recordingBadge: $("recordingBadge"), recordingTime: $("recordingTime"),
  analysisBadge: $("analysisBadge"), recordButton: $("recordButton"), flipButton: $("flipButton"),
  newCaptureButton: $("newCaptureButton"), resultSummary: $("resultSummary"),
  reactionTab: $("reactionTab"), viewPanel: $("viewPanel"), reactionPanel: $("reactionPanel"),
  resultRearVideo: $("resultRearVideo"), resultFrontVideo: $("resultFrontVideo"),
  viewStage: $("viewStage"), heatmapCanvas: $("heatmapCanvas"), heatmapMode: $("heatmapMode"),
  timelineCanvas: $("timelineCanvas"), metricTracked: $("metricTracked"),
  metricPositive: $("metricPositive"), metricZone: $("metricZone"),
  reactionUnavailable: $("reactionUnavailable"), reactionAvailable: $("reactionAvailable"),
  reactionCanvas: $("reactionCanvas"), playReactionButton: $("playReactionButton"),
  exportReactionButton: $("exportReactionButton"), exportStatus: $("exportStatus"),
  downloadVideoButton: $("downloadVideoButton"), downloadDataButton: $("downloadDataButton"),
};

let rearStream = null;
let frontStream = null;
let rearRecorder = null;
let frontRecorder = null;
let rearChunks = [];
let frontChunks = [];
let rearBlob = null;
let frontBlob = null;
let faceLandmarker = null;
let analysisRunning = false;
let analysisRaf = 0;
let lastAnalysisAt = 0;
let latestMetrics = null;
let recording = false;
let recordStart = 0;
let recordTimer = 0;
let samples = [];
let calibrationModel = null;
let calibrationCollect = null;
let frontMirror = true;
let reactionRaf = 0;
let rearUrl = "";
let frontUrl = "";

function showScreen(name) {
  els.setupScreen.classList.toggle("hidden", name !== "setup");
  els.cameraScreen.classList.toggle("hidden", name !== "camera");
  els.resultsScreen.classList.toggle("hidden", name !== "results");
}

function setSetupStatus(message, error = false) {
  els.setupStatus.textContent = message;
  els.setupStatus.style.color = error ? "#ff8d84" : "";
}

function updateConsent() {
  els.prepareButton.disabled = !els.consentAnalysis.checked;
  setSetupStatus(els.consentAnalysis.checked
    ? "映像と分析値はこの端末内だけで処理されます"
    : "同意項目を確認してください");
}

async function loadFaceModel() {
  if (faceLandmarker) return;
  els.analysisBadge.querySelector("span").textContent = "表情モデルを読込中";
  const { FilesetResolver, FaceLandmarker } = await import(MEDIAPIPE_MODULE);
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

async function requestCamera(constraints, fallback) {
  try {
    return await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
  } catch (error) {
    if (!fallback) throw error;
    return navigator.mediaDevices.getUserMedia({ video: fallback, audio: false });
  }
}

async function prepareCameras() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setSetupStatus("このブラウザはカメラ録画に対応していません。最新版のSafariまたはChromeをお試しください。", true);
    return;
  }

  els.prepareButton.disabled = true;
  setSetupStatus("外カメと内カメを準備しています…");
  showScreen("camera");
  try {
    rearStream = await requestCamera(
      { facingMode: { exact: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    );
    els.rearPreview.srcObject = rearStream;
    await els.rearPreview.play();

    try {
      frontStream = await requestCamera(
        { facingMode: { exact: "user" }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 24 } },
        null,
      );
      els.frontPreview.srcObject = frontStream;
      await els.frontPreview.play();
      els.cameraModeBadge.textContent = "DUAL";
      els.analysisBadge.querySelector("span").textContent = "表情モデルを読込中";
      try {
        await loadFaceModel();
        startAnalysisLoop();
        els.analysisBadge.querySelector("span").textContent = "表情・視線を端末内解析";
        els.recordButton.disabled = false;
      } catch (modelError) {
        console.error(modelError);
        els.analysisBadge.querySelector("span").textContent = "モデル読込失敗・外カメのみ";
        els.cameraHint.textContent = "表情モデルを読み込めませんでした。外カメ動画だけ撮影できます";
        els.recordButton.disabled = false;
      }
    } catch (frontError) {
      console.warn("Dual camera unavailable", frontError);
      frontStream = null;
      els.cameraModeBadge.textContent = "SCENE ONLY";
      els.analysisBadge.querySelector("span").textContent = "この端末は前後同時カメラ非対応";
      els.cameraHint.textContent = "外カメ動画のみ撮影できます。表情・視線は記録されません";
      els.calibrateButton.disabled = true;
      els.recordButton.disabled = false;
    }
  } catch (error) {
    console.error(error);
    stopAllStreams();
    showScreen("setup");
    setSetupStatus(cameraErrorMessage(error), true);
    els.prepareButton.disabled = false;
  }
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") return "カメラが許可されていません。ブラウザのサイト設定でカメラを許可してください。";
  if (error?.name === "NotFoundError") return "利用できるカメラが見つかりません。";
  if (error?.name === "NotReadableError") return "別のアプリがカメラを使用中の可能性があります。";
  return `カメラを開始できませんでした（${error?.name || "unknown"}）`;
}

function stopAllStreams() {
  analysisRunning = false;
  cancelAnimationFrame(analysisRaf);
  [rearStream, frontStream].forEach((stream) => stream?.getTracks().forEach((track) => track.stop()));
  rearStream = null;
  frontStream = null;
  els.rearPreview.srcObject = null;
  els.frontPreview.srcObject = null;
}

function startAnalysisLoop() {
  analysisRunning = true;
  const loop = () => {
    if (!analysisRunning) return;
    analysisRaf = requestAnimationFrame(loop);
    if (!faceLandmarker || els.frontPreview.readyState < 2) return;
    const now = performance.now();
    if (now - lastAnalysisAt < ANALYSIS_INTERVAL_MS) return;
    lastAnalysisAt = now;
    try {
      const result = faceLandmarker.detectForVideo(els.frontPreview, now);
      latestMetrics = computeFaceMetrics(result);
      updateAnalysisBadge(latestMetrics);
      if (calibrationCollect && latestMetrics?.rawGazeX != null) {
        calibrationCollect.push({ x: latestMetrics.rawGazeX, y: latestMetrics.rawGazeY });
      }
      if (recording) sampleMetrics(now, latestMetrics);
    } catch (error) {
      console.warn("Face analysis skipped", error);
    }
  };
  loop();
}

function blendshapeMap(result) {
  const categories = result?.faceBlendshapes?.[0]?.categories;
  if (!categories) return null;
  return Object.fromEntries(categories.map((item) => [item.categoryName, item.score]));
}

function computeFaceMetrics(result) {
  const bs = blendshapeMap(result);
  const lm = result?.faceLandmarks?.[0];
  if (!bs || !lm) return { faceDetected: false };
  const blink = ((bs.eyeBlinkLeft ?? 0) + (bs.eyeBlinkRight ?? 0)) / 2;
  const eyeOpen = 1 - blink;
  const smile = ((bs.mouthSmileLeft ?? 0) + (bs.mouthSmileRight ?? 0)) / 2;
  const furrow = ((bs.browDownLeft ?? 0) + (bs.browDownRight ?? 0)) / 2;
  const browRaise = bs.browInnerUp ?? 0;
  const nose = lm[1], eyeL = lm[33], eyeR = lm[263];
  const midX = (eyeL.x + eyeR.x) / 2;
  const midY = (eyeL.y + eyeR.y) / 2;
  const io = Math.max(Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y), 1e-6);
  const yaw = (nose.x - midX) / io;
  const pitch = (nose.y - midY) / io;
  let rawGazeX = null, rawGazeY = null;
  if (eyeOpen > 0.32 && lm[473]) {
    const ratio = (v, a, b) => Math.abs(b - a) > 1e-6 ? (v - a) / (b - a) : 0.5;
    const hx = (ratio(lm[468].x, lm[33].x, lm[133].x) + ratio(lm[473].x, lm[362].x, lm[263].x)) / 2;
    const vy = (ratio(lm[468].y, lm[159].y, lm[145].y) + ratio(lm[473].y, lm[386].y, lm[374].y)) / 2;
    rawGazeX = clamp(-((hx - 0.5) * 4.4 + 1.2 * yaw), -2, 2);
    rawGazeY = clamp((vy - 0.5) * 4.4 + 1.1 * (pitch - 0.48), -2, 2);
  }
  return {
    faceDetected: true, smile, furrow, browRaise, eyeOpen,
    valence: smile - furrow, yaw, pitch, rawGazeX, rawGazeY,
    attention: Math.abs(yaw) < 0.35 && eyeOpen > 0.3 ? 1 : 0,
  };
}

function updateAnalysisBadge(metrics) {
  const span = els.analysisBadge.querySelector("span");
  if (!metrics?.faceDetected) {
    span.textContent = "顔を画面側へ向けてください";
    return;
  }
  span.textContent = metrics.rawGazeX == null ? "目を確認中" : "表情・視線を端末内解析";
}

function mapGaze(rawX, rawY) {
  if (rawX == null || rawY == null) return null;
  if (calibrationModel) {
    return {
      x: clamp(calibrationModel.x[0] * rawX + calibrationModel.x[1] * rawY + calibrationModel.x[2], 0, 1),
      y: clamp(calibrationModel.y[0] * rawX + calibrationModel.y[1] * rawY + calibrationModel.y[2], 0, 1),
      calibrated: true,
    };
  }
  return { x: clamp((rawX + 1) / 2, 0, 1), y: clamp((rawY + 1) / 2, 0, 1), calibrated: false };
}

function sampleMetrics(now, metrics) {
  const gaze = metrics?.faceDetected ? mapGaze(metrics.rawGazeX, metrics.rawGazeY) : null;
  samples.push({
    elapsed_ms: Math.round(now - recordStart),
    face_detected: metrics?.faceDetected ? 1 : 0,
    gaze_x: round(gaze?.x), gaze_y: round(gaze?.y), gaze_calibrated: gaze?.calibrated ? 1 : 0,
    gaze_zone: gaze ? gazeZone(gaze.x, gaze.y) : "",
    attention: metrics?.attention ?? 0,
    smile: round(metrics?.smile), brow_furrow: round(metrics?.furrow),
    brow_raise: round(metrics?.browRaise), eye_open: round(metrics?.eyeOpen),
    valence: round(metrics?.valence), yaw_proxy: round(metrics?.yaw), pitch_proxy: round(metrics?.pitch),
  });
}

async function runCalibration() {
  if (!frontStream || !faceLandmarker || recording) return;
  els.calibrateButton.disabled = true;
  els.recordButton.disabled = true;
  els.calibrationLayer.classList.remove("hidden");
  const observations = [];
  try {
    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
      const point = CALIBRATION_POINTS[i];
      els.calibrationTarget.style.left = `${point.x * 100}%`;
      els.calibrationTarget.style.top = `${point.y * 100}%`;
      els.calibrationProgress.textContent = `${i + 1} / ${CALIBRATION_POINTS.length}`;
      calibrationCollect = [];
      await delay(450);
      calibrationCollect = [];
      await delay(900);
      if (calibrationCollect.length < 2) throw new Error("視線を検出できませんでした");
      observations.push({ rawX: median(calibrationCollect.map((p) => p.x)), rawY: median(calibrationCollect.map((p) => p.y)), targetX: point.x, targetY: point.y });
    }
    calibrationModel = fitCalibration(observations);
    els.cameraHint.textContent = "視線調整が完了しました。録画を開始できます";
    els.calibrateButton.innerHTML = "<span>✓</span>調整済み";
  } catch (error) {
    console.warn(error);
    calibrationModel = null;
    els.cameraHint.textContent = "視線調整を完了できませんでした。顔を画面側に向けて再度お試しください";
  } finally {
    calibrationCollect = null;
    els.calibrationLayer.classList.add("hidden");
    els.calibrateButton.disabled = false;
    els.recordButton.disabled = false;
  }
}

function fitCalibration(observations) {
  const features = observations.map((o) => [o.rawX, o.rawY, 1]);
  const xtx = Array.from({ length: 3 }, (_, r) => Array.from({ length: 3 }, (_, c) => features.reduce((s, f) => s + f[r] * f[c], 0)));
  const solveFor = (key) => {
    const xty = Array.from({ length: 3 }, (_, r) => features.reduce((s, f, i) => s + f[r] * observations[i][key], 0));
    return solve3(xtx.map((row) => [...row]), xty);
  };
  return { x: solveFor("targetX"), y: solveFor("targetY") };
}

function solve3(matrix, vector) {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    if (Math.abs(a[col][col]) < 1e-7) throw new Error("視線調整の差が不足しています");
    const scale = a[col][col];
    for (let c = col; c < 4; c++) a[col][c] /= scale;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let c = col; c < 4; c++) a[row][c] -= factor * a[col][c];
    }
  }
  return a.map((row) => row[3]);
}

function supportedMime() {
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function makeRecorder(stream, chunks) {
  const mimeType = supportedMime();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 4_000_000 } : undefined);
  recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data); };
  return recorder;
}

async function startRecording() {
  if (!rearStream || recording) return;
  rearChunks = [];
  frontChunks = [];
  samples = [];
  rearRecorder = makeRecorder(rearStream, rearChunks);
  frontRecorder = null;
  if (frontStream && els.saveReactionVideo.checked) {
    try {
      frontRecorder = makeRecorder(frontStream, frontChunks);
    } catch (error) {
      console.warn("Front video recording unavailable", error);
      els.cameraHint.textContent = "この端末では表情映像を保存できないため、数値解析だけ記録します";
    }
  }
  recordStart = performance.now();
  recording = true;
  rearRecorder.start(500);
  frontRecorder?.start(500);
  els.recordButton.classList.add("recording");
  els.recordingBadge.classList.remove("hidden");
  els.cameraHint.textContent = frontRecorder ? "表情映像も端末内に保存しています" : "表情は数値だけ記録し、内カメ映像は保存しません";
  els.calibrateButton.disabled = true;
  recordTimer = window.setInterval(updateRecordTime, 250);
  updateRecordTime();
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(recordTimer);
  els.recordButton.disabled = true;
  els.cameraHint.textContent = "映像を端末内でまとめています…";
  const rearType = rearRecorder?.mimeType || "video/webm";
  const frontType = frontRecorder?.mimeType || "video/webm";
  await Promise.all([stopRecorder(rearRecorder), stopRecorder(frontRecorder)]);
  rearBlob = new Blob(rearChunks, { type: rearType });
  frontBlob = frontChunks.length ? new Blob(frontChunks, { type: frontType }) : null;
  stopAllStreams();
  prepareResults();
}

function stopRecorder(recorder) {
  if (!recorder || recorder.state === "inactive") return Promise.resolve();
  return new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.stop();
  });
}

function updateRecordTime() {
  const seconds = Math.floor((performance.now() - recordStart) / 1000);
  els.recordingTime.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function prepareResults() {
  if (rearUrl) URL.revokeObjectURL(rearUrl);
  if (frontUrl) URL.revokeObjectURL(frontUrl);
  rearUrl = URL.createObjectURL(rearBlob);
  frontUrl = frontBlob ? URL.createObjectURL(frontBlob) : "";
  els.resultRearVideo.src = rearUrl;
  els.resultFrontVideo.src = frontUrl;
  els.reactionUnavailable.classList.toggle("hidden", !!frontBlob);
  els.reactionAvailable.classList.toggle("hidden", !frontBlob);
  els.reactionTab.disabled = !frontBlob;
  showScreen("results");
  summarizeResults();
  requestAnimationFrame(() => {
    resizeHeatmap();
    drawHeatmap();
    drawTimeline();
    if (frontBlob) drawReactionFrame();
  });
}

function summarizeResults() {
  const total = samples.length;
  const tracked = samples.filter((s) => s.face_detected && s.gaze_x !== "");
  const positive = samples.filter((s) => number(s.smile) >= 0.35);
  const zoneCounts = {};
  tracked.forEach((s) => { zoneCounts[s.gaze_zone] = (zoneCounts[s.gaze_zone] || 0) + 1; });
  const topZone = Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
  els.metricTracked.textContent = total ? `${Math.round(tracked.length / total * 100)}%` : "—";
  els.metricPositive.textContent = total ? `${Math.round(positive.length / total * 100)}%` : "—";
  els.metricZone.textContent = zoneLabel(topZone);
  els.resultSummary.textContent = tracked.length
    ? `${Math.round((samples.at(-1)?.elapsed_ms || 0) / 1000)}秒の撮影から、${tracked.length}点の視線・表情データを同期しました。`
    : "外カメ映像を保存しました。この端末では視線データを十分に取得できませんでした。";
}

function resizeHeatmap() {
  const rect = els.viewStage.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  els.heatmapCanvas.width = Math.round(rect.width * dpr);
  els.heatmapCanvas.height = Math.round(rect.height * dpr);
  els.heatmapCanvas.style.width = `${rect.width}px`;
  els.heatmapCanvas.style.height = `${rect.height}px`;
}

function displayedVideoRect(video, canvas) {
  const cw = canvas.width, ch = canvas.height;
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const scale = Math.min(cw / vw, ch / vh);
  const width = vw * scale, height = vh * scale;
  return { x: (cw - width) / 2, y: (ch - height) / 2, width, height };
}

function drawHeatmap() {
  const canvas = els.heatmapCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const mode = els.heatmapMode.value;
  if (mode === "off" || !samples.length) return;
  const t = els.resultRearVideo.currentTime * 1000;
  const visible = samples.filter((s) => s.gaze_x !== "" && (mode === "overall" || Math.abs(s.elapsed_ms - t) <= 1200));
  const rect = displayedVideoRect(els.resultRearVideo, canvas);
  ctx.globalCompositeOperation = "lighter";
  for (const sample of visible) {
    const x = rect.x + number(sample.gaze_x) * rect.width;
    const y = rect.y + number(sample.gaze_y) * rect.height;
    const radius = Math.max(28, rect.width * (mode === "overall" ? .065 : .09));
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, mode === "overall" ? "rgba(255,40,20,.16)" : "rgba(255,40,20,.64)");
    gradient.addColorStop(.34, mode === "overall" ? "rgba(255,174,20,.10)" : "rgba(255,174,20,.42)");
    gradient.addColorStop(1, "rgba(255,230,40,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  ctx.globalCompositeOperation = "source-over";
}

function drawTimeline() {
  const canvas = els.timelineCanvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { ctx.beginPath(); ctx.moveTo(0, h * i / 4); ctx.lineTo(w, h * i / 4); ctx.stroke(); }
  const duration = Math.max(samples.at(-1)?.elapsed_ms || 1, 1);
  drawSeries(ctx, samples, duration, w, h, "valence", "#e5ff3f", (v) => .5 - number(v) * .35);
  drawSeries(ctx, samples, duration, w, h, "smile", "#ff6f61", (v) => .92 - number(v) * .72);
  const cursorX = clamp((els.resultRearVideo.currentTime * 1000) / duration, 0, 1) * w;
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cursorX, 0); ctx.lineTo(cursorX, h); ctx.stroke();
}

function drawSeries(ctx, rows, duration, w, h, key, color, yFn) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, devicePixelRatio || 1);
  ctx.beginPath();
  let started = false;
  for (const row of rows) {
    if (row[key] === "") continue;
    const x = row.elapsed_ms / duration * w;
    const y = clamp(yFn(row[key]), .04, .96) * h;
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawReactionFrame() {
  cancelAnimationFrame(reactionRaf);
  const canvas = els.reactionCanvas;
  const ctx = canvas.getContext("2d");
  const front = els.resultFrontVideo;
  const rear = els.resultRearVideo;
  const loop = () => {
    drawReactionComposite(ctx, canvas, front, rear);
    if (!front.paused || !rear.paused) reactionRaf = requestAnimationFrame(loop);
  };
  loop();
}

function drawReactionComposite(ctx, canvas, front, rear) {
  ctx.fillStyle = "#090b10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (front.readyState >= 2) drawCover(ctx, front, 0, 0, canvas.width, canvas.height, true);
  const margin = 30;
  const insetW = Math.round(canvas.width * .42);
  const insetH = Math.round(insetW * 9 / 16);
  const insetX = canvas.width - insetW - margin;
  const insetY = margin + 48;
  ctx.fillStyle = "rgba(0,0,0,.55)";
  roundRect(ctx, insetX - 7, insetY - 7, insetW + 14, insetH + 14, 18);
  ctx.fill();
  if (rear.readyState >= 2) drawCover(ctx, rear, insetX, insetY, insetW, insetH, false);
  const nearest = nearestSample(rear.currentTime * 1000);
  if (nearest?.gaze_x !== "") {
    const gx = insetX + number(nearest.gaze_x) * insetW;
    const gy = insetY + number(nearest.gaze_y) * insetH;
    const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, 32);
    grad.addColorStop(0, "rgba(255,55,30,.9)");
    grad.addColorStop(.35, "rgba(255,190,35,.5)");
    grad.addColorStop(1, "rgba(255,220,30,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(gx - 32, gy - 32, 64, 64);
  }
  ctx.fillStyle = "rgba(0,0,0,.48)";
  roundRect(ctx, 24, canvas.height - 88, 212, 50, 25);
  ctx.fill();
  ctx.fillStyle = "#e5ff3f";
  ctx.font = "700 24px system-ui";
  ctx.fillText("ViewPulse", 48, canvas.height - 55);
}

function drawCover(ctx, video, x, y, w, h, mirror) {
  const vw = video.videoWidth || w, vh = video.videoHeight || h;
  const scale = Math.max(w / vw, h / vh);
  const sw = w / scale, sh = h / scale;
  const sx = (vw - sw) / 2, sy = (vh - sh) / 2;
  ctx.save();
  if (mirror) { ctx.translate(x + w, y); ctx.scale(-1, 1); ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h); }
  else ctx.drawImage(video, sx, sy, sw, sh, x, y, w, h);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, radius) {
  const r = Math.min(radius, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function nearestSample(ms) {
  if (!samples.length) return null;
  const index = clamp(Math.round(ms / ANALYSIS_INTERVAL_MS), 0, samples.length - 1);
  let best = samples[index];
  for (let i = Math.max(0, index - 3); i <= Math.min(samples.length - 1, index + 3); i++) {
    if (Math.abs(samples[i].elapsed_ms - ms) < Math.abs(best.elapsed_ms - ms)) best = samples[i];
  }
  return best;
}

async function playReaction() {
  if (!frontBlob) return;
  const front = els.resultFrontVideo, rear = els.resultRearVideo;
  front.currentTime = 0;
  rear.currentTime = 0;
  await Promise.allSettled([front.play(), rear.play()]);
  els.playReactionButton.textContent = "再生中…";
  drawReactionFrame();
  front.addEventListener("ended", () => { els.playReactionButton.textContent = "▶ もう一度再生"; }, { once: true });
}

async function exportReaction() {
  if (!frontBlob || !els.reactionCanvas.captureStream) {
    els.exportStatus.textContent = "このブラウザはリアクション映像の書き出しに対応していません。";
    return;
  }
  els.exportReactionButton.disabled = true;
  els.exportStatus.textContent = "映像の長さと同じ時間をかけて書き出しています…";
  const stream = els.reactionCanvas.captureStream(30);
  const chunks = [];
  const mimeType = supportedMime();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 5_000_000 } : undefined);
  recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data); };
  const finished = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
  els.resultFrontVideo.currentTime = 0;
  els.resultRearVideo.currentTime = 0;
  recorder.start(500);
  await Promise.allSettled([els.resultFrontVideo.play(), els.resultRearVideo.play()]);
  drawReactionFrame();
  await new Promise((resolve) => els.resultFrontVideo.addEventListener("ended", resolve, { once: true }));
  recorder.stop();
  await finished;
  stream.getTracks().forEach((track) => track.stop());
  const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
  downloadBlob(blob, `viewpulse_reaction_${timestamp()}.${extensionFor(blob.type)}`);
  els.exportStatus.textContent = "リアクション映像を書き出しました。";
  els.exportReactionButton.disabled = false;
}

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  els.viewPanel.classList.toggle("hidden", name !== "view");
  els.reactionPanel.classList.toggle("hidden", name !== "reaction");
  if (name === "reaction" && frontBlob) drawReactionFrame();
  if (name === "view") { resizeHeatmap(); drawHeatmap(); drawTimeline(); }
}

function downloadBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function timestamp() { return new Date().toISOString().replace(/[:.]/g, "-"); }
function extensionFor(type) { return type.includes("mp4") ? "mp4" : "webm"; }
function round(value) { return value == null || !Number.isFinite(value) ? "" : Math.round(value * 1000) / 1000; }
function number(value) { const n = Number(value); return Number.isFinite(n) ? n : 0; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function median(values) { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }
function gazeZone(x, y) {
  const col = x < .333 ? "left" : x > .666 ? "right" : "center";
  const row = y < .333 ? "up" : y > .666 ? "down" : "middle";
  return `${col}-${row}`;
}
function zoneLabel(zone) {
  const labels = { "left-up": "左上", "center-up": "中央上", "right-up": "右上", "left-middle": "左", "center-middle": "中央", "right-middle": "右", "left-down": "左下", "center-down": "中央下", "right-down": "右下" };
  return labels[zone] || "—";
}

els.consentAnalysis.addEventListener("change", updateConsent);
els.prepareButton.addEventListener("click", prepareCameras);
els.calibrateButton.addEventListener("click", runCalibration);
els.recordButton.addEventListener("click", () => recording ? stopRecording() : startRecording());
els.closeCameraButton.addEventListener("click", () => {
  if (recording) stopRecording();
  else { stopAllStreams(); showScreen("setup"); els.prepareButton.disabled = !els.consentAnalysis.checked; }
});
els.flipButton.addEventListener("click", () => {
  frontMirror = !frontMirror;
  els.rearPreview.style.transform = frontMirror ? "" : "scaleX(-1)";
});
els.newCaptureButton.addEventListener("click", () => location.reload());
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => { if (!button.disabled) selectTab(button.dataset.tab); }));
els.resultRearVideo.addEventListener("timeupdate", () => { drawHeatmap(); drawTimeline(); });
els.resultRearVideo.addEventListener("loadedmetadata", () => { resizeHeatmap(); drawHeatmap(); drawTimeline(); });
els.heatmapMode.addEventListener("change", drawHeatmap);
els.playReactionButton.addEventListener("click", playReaction);
els.exportReactionButton.addEventListener("click", exportReaction);
els.downloadVideoButton.addEventListener("click", () => rearBlob && downloadBlob(rearBlob, `viewpulse_scene_${timestamp()}.${extensionFor(rearBlob.type)}`));
els.downloadDataButton.addEventListener("click", () => downloadBlob(new Blob([JSON.stringify({ app: "ViewPulse", created_at: new Date().toISOString(), calibration: calibrationModel ? "five-point" : "uncalibrated", samples }, null, 2)], { type: "application/json" }), `viewpulse_data_${timestamp()}.json`));
window.addEventListener("resize", () => { if (!els.resultsScreen.classList.contains("hidden")) { resizeHeatmap(); drawHeatmap(); drawTimeline(); } });
window.addEventListener("pagehide", stopAllStreams);

updateConsent();
