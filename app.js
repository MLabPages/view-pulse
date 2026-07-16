const MEDIAPIPE_MODULE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ANALYSIS_INTERVAL_MS = 200;
const LIBRARY_DB_NAME = "viewpulse-library";
const LIBRARY_DB_VERSION = 1;
const LIBRARY_STORE = "captures";
const CALIBRATION_POINTS = [
  { x: 0.18, y: 0.2 }, { x: 0.82, y: 0.2 }, { x: 0.5, y: 0.5 },
  { x: 0.18, y: 0.75 }, { x: 0.82, y: 0.75 },
];

const $ = (id) => document.getElementById(id);
const els = {
  setupScreen: $("setupScreen"), captureScreen: $("captureScreen"), resultsScreen: $("resultsScreen"), libraryScreen: $("libraryScreen"),
  contentFileInput: $("contentFileInput"), selectedMediaPreview: $("selectedMediaPreview"),
  selectedMediaName: $("selectedMediaName"), selectedMediaMeta: $("selectedMediaMeta"),
  consentAnalysis: $("consentAnalysis"), saveReactionVideo: $("saveReactionVideo"),
  prepareButton: $("prepareButton"), setupStatus: $("setupStatus"), openLibraryButton: $("openLibraryButton"),
  closeLibraryButton: $("closeLibraryButton"), libraryCountBadge: $("libraryCountBadge"),
  libraryGrid: $("libraryGrid"), libraryEmpty: $("libraryEmpty"), storageStatus: $("storageStatus"),
  contentStage: $("contentStage"), contentImage: $("contentImage"), contentVideo: $("contentVideo"),
  frontPreview: $("frontPreview"), contentTypeBadge: $("contentTypeBadge"), closeCaptureButton: $("closeCaptureButton"),
  captureHint: $("captureHint"), calibrationLayer: $("calibrationLayer"), calibrationTarget: $("calibrationTarget"),
  calibrationProgress: $("calibrationProgress"), calibrateButton: $("calibrateButton"),
  recordingBadge: $("recordingBadge"), recordingTime: $("recordingTime"), analysisBadge: $("analysisBadge"),
  recordButton: $("recordButton"), fullscreenButton: $("fullscreenButton"),
  pipModeButton: $("pipModeButton"), hiddenModeButton: $("hiddenModeButton"),
  newCaptureButton: $("newCaptureButton"), resultSummary: $("resultSummary"),
  reactionTab: $("reactionTab"), viewPanel: $("viewPanel"), reactionPanel: $("reactionPanel"),
  resultContentImage: $("resultContentImage"), resultContentVideo: $("resultContentVideo"), resultFrontVideo: $("resultFrontVideo"),
  viewStage: $("viewStage"), heatmapCanvas: $("heatmapCanvas"), heatmapMode: $("heatmapMode"),
  timelineCanvas: $("timelineCanvas"), timelineHelp: $("timelineHelp"), metricTracked: $("metricTracked"),
  metricPositive: $("metricPositive"), metricZone: $("metricZone"),
  reactionUnavailable: $("reactionUnavailable"), reactionAvailable: $("reactionAvailable"),
  reactionCanvas: $("reactionCanvas"), playReactionButton: $("playReactionButton"),
  exportReactionButton: $("exportReactionButton"), exportStatus: $("exportStatus"),
  downloadContentButton: $("downloadContentButton"), downloadDataButton: $("downloadDataButton"),
  shareCaptureButton: $("shareCaptureButton"), saveStatus: $("saveStatus"),
};

let selectedFile = null;
let selectedUrl = "";
let contentBlob = null;
let contentKind = "";
let contentName = "";
let contentMime = "";
let contentDurationMs = 0;
let frontStream = null;
let frontRecorder = null;
let frontChunks = [];
let frontBlob = null;
let faceLandmarker = null;
let analysisRunning = false;
let analysisRaf = 0;
let lastAnalysisAt = 0;
let latestMetrics = null;
let recording = false;
let stopping = false;
let recordStart = 0;
let recordTimer = 0;
let samples = [];
let calibrationModel = null;
let calibrationCollect = null;
let reactionRaf = 0;
let contentResultUrl = "";
let frontResultUrl = "";
let currentCaptureId = "";
let currentCaptureCreatedAt = "";
let libraryObjectUrls = [];
let imageTimelineMs = 0;

function showScreen(name) {
  els.setupScreen.classList.toggle("hidden", name !== "setup");
  els.captureScreen.classList.toggle("hidden", name !== "capture");
  els.resultsScreen.classList.toggle("hidden", name !== "results");
  els.libraryScreen.classList.toggle("hidden", name !== "library");
}

function setSetupStatus(message, error = false) {
  els.setupStatus.textContent = message;
  els.setupStatus.classList.toggle("error", error);
}

function updateReadiness() {
  const ready = !!selectedFile && els.consentAnalysis.checked;
  els.prepareButton.disabled = !ready;
  if (!selectedFile) setSetupStatus("画像または動画を選んでください");
  else if (!els.consentAnalysis.checked) setSetupStatus("端末内解析への同意を確認してください");
  else setSetupStatus("選んだコンテンツと解析値は、この端末内だけで処理されます");
}

function releaseSelectedUrl() {
  if (selectedUrl) URL.revokeObjectURL(selectedUrl);
  selectedUrl = "";
}

async function selectContentFile(file) {
  if (!file || (!file.type.startsWith("image/") && !file.type.startsWith("video/"))) {
    selectedFile = null;
    setSetupStatus("画像または動画ファイルを選んでください", true);
    updateReadiness();
    return;
  }
  releaseSelectedUrl();
  selectedFile = file;
  selectedUrl = URL.createObjectURL(file);
  contentBlob = file;
  contentKind = file.type.startsWith("image/") ? "image" : "video";
  contentName = file.name || `${contentKind}-${Date.now()}`;
  contentMime = file.type || (contentKind === "image" ? "image/jpeg" : "video/webm");
  contentDurationMs = 0;
  els.selectedMediaPreview.replaceChildren();
  const preview = document.createElement(contentKind === "image" ? "img" : "video");
  preview.src = selectedUrl;
  preview.alt = contentKind === "image" ? "選択した画像のプレビュー" : "";
  if (contentKind === "video") {
    preview.muted = true;
    preview.playsInline = true;
    preview.preload = "metadata";
    preview.addEventListener("loadedmetadata", () => {
      contentDurationMs = Number.isFinite(preview.duration) ? Math.round(preview.duration * 1000) : 0;
      updateSelectedMediaMeta();
    }, { once: true });
  }
  els.selectedMediaPreview.append(preview);
  els.selectedMediaName.textContent = contentName;
  updateSelectedMediaMeta();
  updateReadiness();
}

function updateSelectedMediaMeta() {
  const kindLabel = contentKind === "image" ? "画像" : "動画";
  const duration = contentKind === "video" && contentDurationMs ? `・${formatDuration(contentDurationMs)}` : "";
  els.selectedMediaMeta.textContent = `${kindLabel}${duration}・${formatBytes(selectedFile?.size || 0)}・端末内のみ`;
}

async function loadFaceModel() {
  if (faceLandmarker) return;
  els.analysisBadge.querySelector("span").textContent = "表情モデルを読込中";
  const { FilesetResolver, FaceLandmarker } = await import(MEDIAPIPE_MODULE);
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  };
  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
  } catch (error) {
    console.warn("GPU model initialization failed; retrying on CPU", error);
    options.baseOptions.delegate = "CPU";
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
  }
}

async function attachCameraVideo(video, stream) {
  video.srcObject = stream;
  if (video.readyState < 1) {
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 2500);
      video.addEventListener("loadedmetadata", () => { clearTimeout(timeout); resolve(); }, { once: true });
    });
  }
  await video.play();
}

async function prepareCapture() {
  if (!selectedFile || !els.consentAnalysis.checked) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setSetupStatus("このブラウザはカメラ解析または録画に対応していません。最新版のSafari・Chrome・Edgeをお試しください。", true);
    return;
  }
  els.prepareButton.disabled = true;
  setSetupStatus("内カメと表情モデルを準備しています…");
  showScreen("capture");
  mountSelectedContent();
  try {
    frontStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 }, frameRate: { ideal: 24, max: 30 } },
      audio: false,
    });
    await attachCameraVideo(els.frontPreview, frontStream);
    await loadFaceModel();
    startAnalysisLoop();
    els.analysisBadge.querySelector("span").textContent = "表情・視線を端末内解析";
    els.recordButton.disabled = false;
  } catch (error) {
    console.error(error);
    stopAllStreams();
    showScreen("setup");
    setSetupStatus(cameraErrorMessage(error), true);
    els.prepareButton.disabled = false;
  }
}

function mountSelectedContent() {
  const isImage = contentKind === "image";
  els.contentImage.classList.toggle("hidden", !isImage);
  els.contentVideo.classList.toggle("hidden", isImage);
  els.contentTypeBadge.textContent = isImage ? "IMAGE" : "VIDEO";
  if (isImage) {
    els.contentImage.src = selectedUrl;
    els.contentVideo.removeAttribute("src");
    els.contentVideo.load();
  } else {
    els.contentVideo.src = selectedUrl;
    els.contentVideo.currentTime = 0;
    els.contentImage.removeAttribute("src");
  }
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") return "内カメが許可されていません。ブラウザのサイト設定でカメラを許可してください。";
  if (error?.name === "NotFoundError") return "利用できる内カメが見つかりません。";
  if (error?.name === "NotReadableError") return "別のアプリがカメラを使用中の可能性があります。";
  return `内カメを開始できませんでした（${error?.name || "unknown"}）`;
}

function stopAllStreams() {
  analysisRunning = false;
  cancelAnimationFrame(analysisRaf);
  frontStream?.getTracks().forEach((track) => track.stop());
  frontStream = null;
  els.frontPreview.srcObject = null;
  els.contentVideo.pause();
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
  return categories ? Object.fromEntries(categories.map((item) => [item.categoryName, item.score])) : null;
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
  if (eyeOpen > 0.32 && lm[468] && lm[473]) {
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
  if (!metrics?.faceDetected) span.textContent = "顔を画面側へ向けてください";
  else span.textContent = metrics.rawGazeX == null ? "目を確認中" : "表情・視線を端末内解析";
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

function currentSyncMs(now = performance.now()) {
  if (contentKind === "video") return Math.round((els.contentVideo.currentTime || 0) * 1000);
  return Math.max(0, Math.round(now - recordStart));
}

function sampleMetrics(now, metrics) {
  const gaze = metrics?.faceDetected ? mapGaze(metrics.rawGazeX, metrics.rawGazeY) : null;
  samples.push({
    elapsed_ms: Math.max(0, Math.round(now - recordStart)),
    sync_ms: currentSyncMs(now),
    content_kind: contentKind,
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
    els.captureHint.textContent = "視線調整が完了しました。記録を開始できます";
    els.calibrateButton.innerHTML = "<span>✓</span>調整済み";
  } catch (error) {
    console.warn(error);
    calibrationModel = null;
    els.captureHint.textContent = "視線調整を完了できませんでした。顔を画面側に向けて再度お試しください";
  } finally {
    calibrationCollect = null;
    els.calibrationLayer.classList.add("hidden");
    els.calibrateButton.disabled = false;
    els.recordButton.disabled = false;
  }
}

function fitCalibration(observations) {
  const features = observations.map((o) => [o.rawX, o.rawY, 1]);
  const xtx = Array.from({ length: 3 }, (_, r) => Array.from({ length: 3 }, (_, c) => features.reduce((sum, f) => sum + f[r] * f[c], 0)));
  const solveFor = (key) => {
    const xty = Array.from({ length: 3 }, (_, r) => features.reduce((sum, f, i) => sum + f[r] * observations[i][key], 0));
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
  if (!frontStream || recording || stopping) return;
  currentCaptureId = "";
  currentCaptureCreatedAt = "";
  frontChunks = [];
  samples = [];
  imageTimelineMs = 0;
  frontRecorder = null;
  if (els.saveReactionVideo.checked) {
    try {
      frontRecorder = makeRecorder(frontStream, frontChunks);
    } catch (error) {
      console.warn("Reaction video recording unavailable", error);
      els.captureHint.textContent = "この端末では表情映像を保存できないため、数値解析だけ記録します";
    }
  }
  if (contentKind === "video") {
    els.contentVideo.currentTime = 0;
    try { await els.contentVideo.play(); } catch (error) { console.warn("Content playback needs another tap", error); }
  }
  recordStart = performance.now();
  recording = true;
  frontRecorder?.start(500);
  els.recordButton.classList.add("recording");
  els.recordingBadge.classList.remove("hidden");
  els.captureHint.textContent = frontRecorder ? "表情映像も端末内に保存しています" : "表情は数値だけ記録し、内カメ映像は保存しません";
  els.calibrateButton.disabled = true;
  recordTimer = window.setInterval(updateRecordTime, 250);
  updateRecordTime();
}

async function stopRecording() {
  if (!recording || stopping) return;
  stopping = true;
  recording = false;
  clearInterval(recordTimer);
  els.contentVideo.pause();
  els.recordButton.disabled = true;
  els.captureHint.textContent = "記録を端末内でまとめています…";
  const frontType = frontRecorder?.mimeType || "video/webm";
  await stopRecorder(frontRecorder);
  frontBlob = frontChunks.length ? new Blob(frontChunks, { type: frontType }) : null;
  if (!samples.length) sampleMetrics(performance.now(), latestMetrics);
  contentDurationMs = contentKind === "video"
    ? Math.max(contentDurationMs, Math.round((els.contentVideo.duration || 0) * 1000))
    : (samples.at(-1)?.elapsed_ms || 0);
  const thumbnail = await createContentThumbnail();
  stopAllStreams();
  prepareResults();
  els.saveStatus.textContent = "この端末のライブラリへ保存しています…";
  try {
    await saveCurrentCapture(thumbnail);
    els.saveStatus.textContent = "この端末のライブラリに保存しました。外部送信はしていません。";
  } catch (error) {
    console.error("Library save failed", error);
    els.saveStatus.textContent = "端末内ライブラリへ保存できませんでした。下の保存・共有ボタンでデータを残してください。";
  } finally {
    stopping = false;
  }
}

function stopRecorder(recorder) {
  if (!recorder || recorder.state === "inactive") return Promise.resolve();
  return new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.stop();
  });
}

function setPreviewMode(mode) {
  const hidden = mode === "hidden";
  els.frontPreview.classList.toggle("preview-hidden", hidden);
  els.frontPreview.classList.toggle("pip-mode", !hidden);
  els.pipModeButton.classList.toggle("active", !hidden);
  els.hiddenModeButton.classList.toggle("active", hidden);
  els.pipModeButton.setAttribute("aria-pressed", String(!hidden));
  els.hiddenModeButton.setAttribute("aria-pressed", String(hidden));
}

function openLibraryDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(LIBRARY_DB_NAME, LIBRARY_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(LIBRARY_STORE)) {
        const store = db.createObjectStore(LIBRARY_STORE, { keyPath: "id" });
        store.createIndex("created_at", "created_at");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("端末内ライブラリを開けませんでした"));
  });
}

async function libraryRequest(mode, action) {
  const db = await openLibraryDb();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction(LIBRARY_STORE, mode);
      const request = action(transaction.objectStore(LIBRARY_STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      transaction.onabort = () => reject(transaction.error || new Error("保存処理が中断されました"));
    });
  } finally {
    db.close();
  }
}

const libraryGetAll = () => libraryRequest("readonly", (store) => store.getAll());
const libraryGet = (id) => libraryRequest("readonly", (store) => store.get(id));
const libraryPut = (capture) => libraryRequest("readwrite", (store) => store.put(capture));
const libraryDelete = (id) => libraryRequest("readwrite", (store) => store.delete(id));
const libraryCount = () => libraryRequest("readonly", (store) => store.count());

function normalizeCapture(record) {
  const legacy = !record.content_blob && !!record.rear_blob;
  const normalizedKind = record.content_kind || (legacy ? "video" : (record.content_mime?.startsWith("image/") ? "image" : "video"));
  const normalizedSamples = (record.samples || []).map((sample) => ({
    ...sample,
    sync_ms: Number.isFinite(Number(sample.sync_ms)) ? Number(sample.sync_ms) : number(sample.elapsed_ms),
    content_kind: sample.content_kind || normalizedKind,
  }));
  return {
    ...record,
    content_blob: record.content_blob || record.rear_blob || null,
    content_kind: normalizedKind,
    content_name: record.content_name || (legacy ? "旧版で撮影した外カメ動画" : "表示コンテンツ"),
    content_mime: record.content_mime || record.content_blob?.type || record.rear_blob?.type || "video/webm",
    duration_ms: record.duration_ms || normalizedSamples.at(-1)?.elapsed_ms || 0,
    samples: normalizedSamples,
    legacy_capture: legacy,
  };
}

async function createContentThumbnail() {
  const source = contentKind === "image" ? els.contentImage : els.contentVideo;
  const width = source.naturalWidth || source.videoWidth;
  const height = source.naturalHeight || source.videoHeight;
  if (!width || !height) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = Math.max(270, Math.round(canvas.width * height / width));
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.76));
}

function newCaptureId() {
  return crypto.randomUUID?.() || `capture-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function saveCurrentCapture(thumbnail) {
  currentCaptureId = newCaptureId();
  currentCaptureCreatedAt = new Date().toISOString();
  navigator.storage?.persist?.().catch(() => false);
  await libraryPut({
    id: currentCaptureId,
    created_at: currentCaptureCreatedAt,
    duration_ms: samples.at(-1)?.elapsed_ms || 0,
    content_duration_ms: contentDurationMs,
    content_blob: contentBlob,
    content_kind: contentKind,
    content_name: contentName,
    content_mime: contentMime,
    front_blob: frontBlob,
    thumbnail_blob: thumbnail,
    samples,
    calibration_model: calibrationModel,
    version: 2,
  });
  await refreshLibraryBadge();
}

async function refreshLibraryBadge() {
  try { els.libraryCountBadge.textContent = String(await libraryCount()); }
  catch { els.libraryCountBadge.textContent = "—"; }
}

async function renderLibrary() {
  libraryObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  libraryObjectUrls = [];
  els.libraryGrid.replaceChildren();
  let captures = [];
  try {
    captures = (await libraryGetAll()).map(normalizeCapture).sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  } catch (error) {
    els.libraryEmpty.classList.remove("hidden");
    els.libraryEmpty.querySelector("h2").textContent = "端末内ライブラリを開けませんでした";
    els.libraryEmpty.querySelector("p").textContent = "プライベートブラウズやサイトデータ設定をご確認ください。";
    return;
  }
  els.libraryEmpty.classList.toggle("hidden", captures.length > 0);
  els.libraryCountBadge.textContent = String(captures.length);
  for (const capture of captures) els.libraryGrid.append(createLibraryCard(capture));
  updateStorageStatus();
}

function createLibraryCard(capture) {
  const card = document.createElement("article");
  card.className = "library-card";
  const thumb = document.createElement("div");
  thumb.className = "library-thumb";
  thumb.dataset.kind = capture.content_kind;
  if (capture.thumbnail_blob) {
    const img = document.createElement("img");
    const url = URL.createObjectURL(capture.thumbnail_blob);
    libraryObjectUrls.push(url);
    img.src = url;
    img.alt = "表示コンテンツのサムネイル";
    thumb.append(img);
  }
  const meta = document.createElement("div");
  meta.className = "library-meta";
  const title = document.createElement("strong");
  title.textContent = formatCaptureDate(capture.created_at);
  const name = document.createElement("span");
  name.textContent = capture.content_name;
  const detail = document.createElement("small");
  const legacyLabel = capture.legacy_capture ? "・旧版データ互換表示" : "";
  detail.textContent = `${formatDuration(capture.duration_ms)}・${capture.content_kind === "image" ? "画像" : "動画"}・${capture.front_blob ? "表情映像あり" : "数値解析のみ"}${legacyLabel}`;
  meta.append(title, name, detail);
  const actions = document.createElement("div");
  actions.className = "library-card-actions";
  const openButton = document.createElement("button");
  openButton.textContent = "分析を見る";
  openButton.addEventListener("click", () => openLibraryCapture(capture.id));
  const shareButton = document.createElement("button");
  shareButton.textContent = "共有";
  shareButton.addEventListener("click", () => shareStoredCapture(capture));
  const deleteButton = document.createElement("button");
  deleteButton.textContent = "削除";
  deleteButton.className = "danger";
  deleteButton.setAttribute("aria-label", `${title.textContent}を削除`);
  deleteButton.addEventListener("click", () => deleteLibraryCapture(capture.id));
  actions.append(openButton, shareButton, deleteButton);
  card.append(thumb, meta, actions);
  return card;
}

async function openLibraryCapture(id) {
  const stored = await libraryGet(id);
  if (!stored) return;
  const capture = normalizeCapture(stored);
  contentBlob = capture.content_blob;
  contentKind = capture.content_kind;
  contentName = capture.content_name;
  contentMime = capture.content_mime;
  contentDurationMs = capture.content_duration_ms || capture.duration_ms;
  frontBlob = capture.front_blob || null;
  samples = capture.samples;
  calibrationModel = capture.calibration_model || null;
  currentCaptureId = capture.id;
  currentCaptureCreatedAt = capture.created_at;
  prepareResults();
  els.saveStatus.textContent = capture.legacy_capture
    ? "旧版ライブラリの外カメ動画を、表示コンテンツとして互換表示しています。元データは変更していません。"
    : "この端末のライブラリから開いています。外部送信はしていません。";
}

async function deleteLibraryCapture(id) {
  if (!confirm("このコンテンツ、表情映像、分析データを、この端末から削除しますか？")) return;
  await libraryDelete(id);
  await renderLibrary();
}

async function updateStorageStatus() {
  if (!navigator.storage?.estimate) {
    els.storageStatus.textContent = "このブラウザでは保存容量を表示できません";
    return;
  }
  const { usage = 0, quota = 0 } = await navigator.storage.estimate();
  els.storageStatus.textContent = `ブラウザ内使用量 ${formatBytes(usage)} / 上限の目安 ${formatBytes(quota)}`;
}

function updateRecordTime() {
  const elapsed = Math.max(0, performance.now() - recordStart);
  els.recordingTime.textContent = formatDuration(elapsed);
}

function prepareResults() {
  if (!contentBlob) return;
  if (contentResultUrl) URL.revokeObjectURL(contentResultUrl);
  if (frontResultUrl) URL.revokeObjectURL(frontResultUrl);
  contentResultUrl = URL.createObjectURL(contentBlob);
  frontResultUrl = frontBlob ? URL.createObjectURL(frontBlob) : "";
  const isImage = contentKind === "image";
  els.resultContentImage.classList.toggle("hidden", !isImage);
  els.resultContentVideo.classList.toggle("hidden", isImage);
  if (isImage) {
    els.resultContentImage.src = contentResultUrl;
    els.resultContentVideo.removeAttribute("src");
    els.resultContentVideo.load();
  } else {
    els.resultContentVideo.src = contentResultUrl;
    els.resultContentImage.removeAttribute("src");
  }
  els.resultFrontVideo.src = frontResultUrl;
  els.reactionUnavailable.classList.toggle("hidden", !!frontBlob);
  els.reactionAvailable.classList.toggle("hidden", !frontBlob);
  els.reactionTab.disabled = !frontBlob;
  els.timelineHelp.textContent = isImage ? "タイムラインをタップして経過時間を確認できます" : "動画の再生位置と連動します";
  els.downloadContentButton.textContent = isImage ? "表示画像を保存" : "表示動画を保存";
  imageTimelineMs = 0;
  showScreen("results");
  selectTab("view");
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
  const tracked = samples.filter((sample) => sample.face_detected && sample.gaze_x !== "");
  const positive = samples.filter((sample) => number(sample.smile) >= 0.35);
  const zoneCounts = {};
  tracked.forEach((sample) => { zoneCounts[sample.gaze_zone] = (zoneCounts[sample.gaze_zone] || 0) + 1; });
  const topZone = Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
  els.metricTracked.textContent = total ? `${Math.round(tracked.length / total * 100)}%` : "—";
  els.metricPositive.textContent = total ? `${Math.round(positive.length / total * 100)}%` : "—";
  els.metricZone.textContent = zoneLabel(topZone);
  const seconds = Math.round((samples.at(-1)?.elapsed_ms || 0) / 1000);
  els.resultSummary.textContent = tracked.length
    ? `${contentKind === "image" ? "画像" : "動画"}と${seconds}秒間の反応から、${tracked.length}点の視線・表情データを同期しました。`
    : `${contentKind === "image" ? "画像" : "動画"}と反応を保存しました。この記録では視線データを十分に取得できませんでした。`;
}

function resizeHeatmap() {
  const rect = els.viewStage.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  els.heatmapCanvas.width = Math.round(rect.width * dpr);
  els.heatmapCanvas.height = Math.round(rect.height * dpr);
  els.heatmapCanvas.style.width = `${rect.width}px`;
  els.heatmapCanvas.style.height = `${rect.height}px`;
}

function displayedMediaRect(media, canvas) {
  const cw = canvas.width, ch = canvas.height;
  const mw = media.naturalWidth || media.videoWidth || 16;
  const mh = media.naturalHeight || media.videoHeight || 9;
  const scale = Math.min(cw / mw, ch / mh);
  const width = mw * scale, height = mh * scale;
  return { x: (cw - width) / 2, y: (ch - height) / 2, width, height };
}

function resultSyncMs() {
  return contentKind === "video" ? els.resultContentVideo.currentTime * 1000 : imageTimelineMs;
}

function sampleTime(sample) {
  return Number.isFinite(Number(sample.sync_ms)) ? Number(sample.sync_ms) : number(sample.elapsed_ms);
}

function drawHeatmap() {
  const canvas = els.heatmapCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const mode = els.heatmapMode.value;
  if (mode === "off" || !samples.length) return;
  const t = resultSyncMs();
  const visible = samples.filter((sample) => sample.gaze_x !== "" && (mode === "overall" || Math.abs(sampleTime(sample) - t) <= 1200));
  const media = contentKind === "image" ? els.resultContentImage : els.resultContentVideo;
  const rect = displayedMediaRect(media, canvas);
  ctx.globalCompositeOperation = "lighter";
  for (const sample of visible) {
    const x = rect.x + number(sample.gaze_x) * rect.width;
    const y = rect.y + number(sample.gaze_y) * rect.height;
    const radius = Math.max(28, rect.width * (mode === "overall" ? 0.065 : 0.09));
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, mode === "overall" ? "rgba(255,40,20,.16)" : "rgba(255,40,20,.64)");
    gradient.addColorStop(0.34, mode === "overall" ? "rgba(255,174,20,.10)" : "rgba(255,174,20,.42)");
    gradient.addColorStop(1, "rgba(255,230,40,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  ctx.globalCompositeOperation = "source-over";
}

function timelineDuration() {
  return Math.max(contentKind === "video" ? contentDurationMs : (samples.at(-1)?.elapsed_ms || 1), 1);
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
  const duration = timelineDuration();
  drawSeries(ctx, samples, duration, w, h, "valence", "#e5ff3f", (value) => 0.5 - number(value) * 0.35);
  drawSeries(ctx, samples, duration, w, h, "smile", "#ff6f61", (value) => 0.92 - number(value) * 0.72);
  const cursorX = clamp(resultSyncMs() / duration, 0, 1) * w;
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cursorX, 0); ctx.lineTo(cursorX, h); ctx.stroke();
}

function drawSeries(ctx, rows, duration, width, height, key, color, yFn) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, devicePixelRatio || 1);
  ctx.beginPath();
  let started = false;
  for (const row of rows) {
    if (row[key] === "") continue;
    const x = sampleTime(row) / duration * width;
    const y = clamp(yFn(row[key]), 0.04, 0.96) * height;
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function seekFromTimeline(event) {
  const rect = els.timelineCanvas.getBoundingClientRect();
  const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  const ms = ratio * timelineDuration();
  if (contentKind === "video") els.resultContentVideo.currentTime = ms / 1000;
  else imageTimelineMs = ms;
  drawHeatmap();
  drawTimeline();
}

function drawReactionFrame() {
  cancelAnimationFrame(reactionRaf);
  const ctx = els.reactionCanvas.getContext("2d");
  const loop = () => {
    drawReactionComposite(ctx, els.reactionCanvas, els.resultFrontVideo);
    if (!els.resultFrontVideo.paused) reactionRaf = requestAnimationFrame(loop);
  };
  loop();
}

function drawReactionComposite(ctx, canvas, front) {
  ctx.fillStyle = "#090b10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (front.readyState >= 2) drawCover(ctx, front, 0, 0, canvas.width, canvas.height, true);
  const margin = 30;
  const insetW = Math.round(canvas.width * 0.42);
  const insetH = Math.round(insetW * 9 / 16);
  const insetX = canvas.width - insetW - margin;
  const insetY = margin + 48;
  ctx.fillStyle = "rgba(0,0,0,.55)";
  roundRect(ctx, insetX - 7, insetY - 7, insetW + 14, insetH + 14, 18);
  ctx.fill();
  const media = contentKind === "image" ? els.resultContentImage : els.resultContentVideo;
  if ((contentKind === "image" && media.complete) || (contentKind === "video" && media.readyState >= 2)) {
    drawCover(ctx, media, insetX, insetY, insetW, insetH, false);
  }
  const syncMs = contentKind === "video" ? els.resultContentVideo.currentTime * 1000 : els.resultFrontVideo.currentTime * 1000;
  const nearest = nearestSample(syncMs);
  if (nearest?.gaze_x !== "") {
    const gx = insetX + number(nearest.gaze_x) * insetW;
    const gy = insetY + number(nearest.gaze_y) * insetH;
    const gradient = ctx.createRadialGradient(gx, gy, 0, gx, gy, 32);
    gradient.addColorStop(0, "rgba(255,55,30,.9)");
    gradient.addColorStop(0.35, "rgba(255,190,35,.5)");
    gradient.addColorStop(1, "rgba(255,220,30,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(gx - 32, gy - 32, 64, 64);
  }
  ctx.fillStyle = "rgba(0,0,0,.48)";
  roundRect(ctx, 24, canvas.height - 88, 212, 50, 25);
  ctx.fill();
  ctx.fillStyle = "#e5ff3f";
  ctx.font = "700 24px system-ui";
  ctx.fillText("ViewPulse", 48, canvas.height - 55);
}

function drawCover(ctx, media, x, y, width, height, mirror) {
  const mediaWidth = media.naturalWidth || media.videoWidth || width;
  const mediaHeight = media.naturalHeight || media.videoHeight || height;
  const scale = Math.max(width / mediaWidth, height / mediaHeight);
  const sourceWidth = width / scale, sourceHeight = height / scale;
  const sourceX = (mediaWidth - sourceWidth) / 2, sourceY = (mediaHeight - sourceHeight) / 2;
  ctx.save();
  if (mirror) {
    ctx.translate(x + width, y);
    ctx.scale(-1, 1);
    ctx.drawImage(media, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, width, height);
  } else {
    ctx.drawImage(media, sourceX, sourceY, sourceWidth, sourceHeight, x, y, width, height);
  }
  ctx.restore();
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function nearestSample(ms) {
  if (!samples.length) return null;
  let best = samples[0];
  for (const sample of samples) {
    if (Math.abs(sampleTime(sample) - ms) < Math.abs(sampleTime(best) - ms)) best = sample;
  }
  return best;
}

async function playReaction() {
  if (!frontBlob) return;
  els.resultFrontVideo.currentTime = 0;
  if (contentKind === "video") {
    els.resultContentVideo.currentTime = 0;
    await Promise.allSettled([els.resultFrontVideo.play(), els.resultContentVideo.play()]);
  } else {
    await els.resultFrontVideo.play().catch(() => {});
  }
  els.playReactionButton.textContent = "再生中…";
  drawReactionFrame();
  els.resultFrontVideo.addEventListener("ended", () => {
    els.resultContentVideo.pause();
    els.playReactionButton.textContent = "▶ もう一度再生";
  }, { once: true });
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
  if (contentKind === "video") els.resultContentVideo.currentTime = 0;
  recorder.start(500);
  await Promise.allSettled([els.resultFrontVideo.play(), contentKind === "video" ? els.resultContentVideo.play() : Promise.resolve()]);
  drawReactionFrame();
  await new Promise((resolve) => els.resultFrontVideo.addEventListener("ended", resolve, { once: true }));
  recorder.stop();
  await finished;
  stream.getTracks().forEach((track) => track.stop());
  const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
  downloadBlob(blob, `viewpulse_reaction_${timestamp()}.${extensionForMime(blob.type)}`);
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
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

function captureDataBlob(capture) {
  return new Blob([JSON.stringify({
    app: "ViewPulse",
    schema_version: 2,
    capture_id: capture.id || "",
    created_at: capture.created_at || new Date().toISOString(),
    content: {
      kind: capture.content_kind || contentKind,
      name: capture.content_name || contentName,
      mime: capture.content_mime || contentMime,
      duration_ms: capture.content_duration_ms || contentDurationMs,
    },
    synchronization: capture.content_kind === "image" ? "elapsed_ms" : "content_playback_ms",
    calibration: capture.calibration_model ? "five-point" : "uncalibrated",
    samples: capture.samples || [],
  }, null, 2)], { type: "application/json" });
}

function extensionForMime(type, kind = "video") {
  const subtype = String(type || "").split("/")[1]?.split(/[;+]/)[0];
  if (subtype && /^[a-z0-9]+$/i.test(subtype)) return subtype === "quicktime" ? "mov" : subtype;
  return kind === "image" ? "jpg" : "webm";
}

function captureShareFiles(rawCapture) {
  const capture = normalizeCapture(rawCapture);
  const stem = `viewpulse_${String(capture.created_at || new Date().toISOString()).replace(/[:.]/g, "-")}`;
  const contentExtension = extensionForMime(capture.content_mime || capture.content_blob?.type, capture.content_kind);
  const files = [new File([capture.content_blob], `${stem}_content.${contentExtension}`, { type: capture.content_mime || capture.content_blob?.type })];
  if (capture.front_blob) files.push(new File([capture.front_blob], `${stem}_reaction-source.${extensionForMime(capture.front_blob.type)}`, { type: capture.front_blob.type }));
  const dataBlob = captureDataBlob(capture);
  files.push(new File([dataBlob], `${stem}_analysis.json`, { type: "application/json" }));
  return files;
}

async function shareStoredCapture(rawCapture) {
  const capture = normalizeCapture(rawCapture);
  if (!capture?.content_blob) return;
  const files = captureShareFiles(capture);
  try {
    if (navigator.share) {
      if (!navigator.canShare || navigator.canShare({ files })) {
        await navigator.share({ title: "ViewPulseの記録", text: "表示したコンテンツと、その瞬間の反応データです。", files });
        return;
      }
      if (navigator.canShare({ files: [files[0]] })) {
        await navigator.share({ title: "ViewPulseの記録", text: "ViewPulseで表示したコンテンツです。分析値は端末内ライブラリに残っています。", files: [files[0]] });
        resultOrLibraryStatus("この共有先は複数ファイル非対応のため、コンテンツだけ共有しました。分析値は端末内に残っています。");
        return;
      }
    }
    files.forEach((file) => downloadBlob(file, file.name));
    resultOrLibraryStatus("共有機能に対応していないため、コンテンツと分析データを端末へ保存しました。");
  } catch (error) {
    if (error?.name !== "AbortError") resultOrLibraryStatus("共有を完了できませんでした。端末の空き容量や共有先をご確認ください。");
  }
}

function resultOrLibraryStatus(message) {
  const target = els.resultsScreen.classList.contains("hidden") ? els.storageStatus : els.saveStatus;
  target.textContent = message;
}

function currentCapture() {
  return {
    id: currentCaptureId,
    created_at: currentCaptureCreatedAt,
    content_blob: contentBlob,
    content_kind: contentKind,
    content_name: contentName,
    content_mime: contentMime,
    content_duration_ms: contentDurationMs,
    front_blob: frontBlob,
    samples,
    calibration_model: calibrationModel,
    version: 2,
  };
}

function timestamp() { return new Date().toISOString().replace(/[:.]/g, "-"); }
function formatDuration(ms) {
  const seconds = Math.max(0, Math.round(number(ms) / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function formatCaptureDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "記録日時不明" : new Intl.DateTimeFormat("ja-JP", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}
function formatBytes(value) {
  const bytes = Math.max(0, number(value));
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}
function round(value) { return value == null || !Number.isFinite(value) ? "" : Math.round(value * 1000) / 1000; }
function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : 0; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function median(values) { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }
function gazeZone(x, y) {
  const col = x < 0.333 ? "left" : x > 0.666 ? "right" : "center";
  const row = y < 0.333 ? "up" : y > 0.666 ? "down" : "middle";
  return `${col}-${row}`;
}
function zoneLabel(zone) {
  const labels = { "left-up": "左上", "center-up": "中央上", "right-up": "右上", "left-middle": "左", "center-middle": "中央", "right-middle": "右", "left-down": "左下", "center-down": "中央下", "right-down": "右下" };
  return labels[zone] || "—";
}

els.contentFileInput.addEventListener("change", () => selectContentFile(els.contentFileInput.files?.[0]));
els.consentAnalysis.addEventListener("change", updateReadiness);
els.prepareButton.addEventListener("click", prepareCapture);
els.openLibraryButton.addEventListener("click", async () => { showScreen("library"); await renderLibrary(); });
els.closeLibraryButton.addEventListener("click", () => showScreen("setup"));
els.calibrateButton.addEventListener("click", runCalibration);
els.recordButton.addEventListener("click", () => recording ? stopRecording() : startRecording());
els.closeCaptureButton.addEventListener("click", () => {
  if (recording) stopRecording();
  else { stopAllStreams(); showScreen("setup"); updateReadiness(); }
});
els.pipModeButton.addEventListener("click", () => setPreviewMode("pip"));
els.hiddenModeButton.addEventListener("click", () => setPreviewMode("hidden"));
els.fullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await els.captureScreen.requestFullscreen();
  } catch (error) { console.warn("Fullscreen unavailable", error); }
});
els.contentVideo.addEventListener("loadedmetadata", () => {
  contentDurationMs = Number.isFinite(els.contentVideo.duration) ? Math.round(els.contentVideo.duration * 1000) : contentDurationMs;
});
els.contentVideo.addEventListener("ended", () => { if (recording) stopRecording(); });
els.newCaptureButton.addEventListener("click", () => location.reload());
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => { if (!button.disabled) selectTab(button.dataset.tab); }));
els.resultContentVideo.addEventListener("timeupdate", () => { drawHeatmap(); drawTimeline(); });
els.resultContentVideo.addEventListener("loadedmetadata", () => { resizeHeatmap(); drawHeatmap(); drawTimeline(); });
els.resultContentImage.addEventListener("load", () => { resizeHeatmap(); drawHeatmap(); });
els.heatmapMode.addEventListener("change", drawHeatmap);
els.timelineCanvas.addEventListener("click", seekFromTimeline);
els.playReactionButton.addEventListener("click", playReaction);
els.exportReactionButton.addEventListener("click", exportReaction);
els.shareCaptureButton.addEventListener("click", () => shareStoredCapture(currentCapture()));
els.downloadContentButton.addEventListener("click", () => contentBlob && downloadBlob(contentBlob, `viewpulse_content_${timestamp()}.${extensionForMime(contentMime || contentBlob.type, contentKind)}`));
els.downloadDataButton.addEventListener("click", () => downloadBlob(captureDataBlob(currentCapture()), `viewpulse_data_${timestamp()}.json`));
window.addEventListener("resize", () => { if (!els.resultsScreen.classList.contains("hidden")) { resizeHeatmap(); drawHeatmap(); drawTimeline(); } });
window.addEventListener("pagehide", stopAllStreams);

setPreviewMode("pip");
updateReadiness();
refreshLibraryBadge();
