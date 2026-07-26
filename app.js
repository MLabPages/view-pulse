import { extractYouTubeVideoId, findSharedYouTubeUrl } from "./youtube-url.mjs";

const MEDIAPIPE_MODULE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ANALYSIS_INTERVAL_MS = 200;
const SPECIALIZED_GAZE_INTERVAL_MS = 160;
const SPECIALIZED_GAZE_MAX_AGE_MS = 1200;
const SPECIALIZED_GAZE_INIT_TIMEOUT_MS = 45000;
const CALIBRATION_MIN_SAMPLES = 2;
const CALIBRATION_POINT_TIMEOUT_MS = 10000;
const CALIBRATION_SPREAD_LIMIT = 0.26;
const CALIBRATION_SAMPLE_TIMEOUT_MS = 6000;
const CALIBRATION_FIT_TIMEOUT_MS = 25000;
const SPECIALIZED_GAZE_CAPTURE_WIDTH = 640;
const SPECIALIZED_GAZE_WORKER_URL = new URL("./vendor/webeyetrack/webeyetrack.worker.js", import.meta.url);
const SPECIALIZED_GAZE_MODEL_URL = new URL("./web/model.json", import.meta.url);
const LIBRARY_DB_NAME = "viewpulse-library";
const LIBRARY_DB_VERSION = 1;
const LIBRARY_STORE = "captures";
const CALIBRATION_POINTS = [
  { x: 0.15, y: 0.15 }, { x: 0.5, y: 0.15 }, { x: 0.85, y: 0.15 },
  { x: 0.15, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.85, y: 0.5 },
  { x: 0.15, y: 0.85 }, { x: 0.5, y: 0.85 }, { x: 0.85, y: 0.85 },
];
const CALIBRATION_VALIDATION_POINTS = [
  { x: 0.32, y: 0.32 }, { x: 0.68, y: 0.32 },
  { x: 0.32, y: 0.68 }, { x: 0.68, y: 0.68 },
];
// Measured after the model has been adapted, to remove any remaining
// systematic shrink or offset before the untouched validation points run.
const CALIBRATION_CORRECTION_POINTS = [
  { x: 0.15, y: 0.15 }, { x: 0.85, y: 0.15 },
  { x: 0.5, y: 0.5 },
  { x: 0.15, y: 0.85 }, { x: 0.85, y: 0.85 },
];
const HEATMAP_MOMENT_WINDOW_MS = 500;
const MAX_VALIDATION_MEAN_DIAGONAL_RATIO = 0.12;
const MAX_VALIDATION_POINT_DIAGONAL_RATIO = 0.18;

const $ = (id) => document.getElementById(id);
const els = {
  setupScreen: $("setupScreen"), captureScreen: $("captureScreen"), resultsScreen: $("resultsScreen"), libraryScreen: $("libraryScreen"),
  contentFileInput: $("contentFileInput"), selectedMediaPreview: $("selectedMediaPreview"),
  selectedMediaName: $("selectedMediaName"), selectedMediaMeta: $("selectedMediaMeta"),
  youtubeUrlInput: $("youtubeUrlInput"), youtubeStatus: $("youtubeStatus"), loadYoutubeButton: $("loadYoutubeButton"),
  consentAnalysis: $("consentAnalysis"), saveReactionVideo: $("saveReactionVideo"),
  prepareButton: $("prepareButton"), setupStatus: $("setupStatus"), openLibraryButton: $("openLibraryButton"),
  closeLibraryButton: $("closeLibraryButton"), libraryCountBadge: $("libraryCountBadge"),
  libraryGrid: $("libraryGrid"), libraryEmpty: $("libraryEmpty"), storageStatus: $("storageStatus"),
  contentStage: $("contentStage"), contentImage: $("contentImage"), contentVideo: $("contentVideo"),
  captureYoutubeWrap: $("captureYoutubeWrap"),
  frontPreview: $("frontPreview"), contentTypeBadge: $("contentTypeBadge"), closeCaptureButton: $("closeCaptureButton"),
  captureHint: $("captureHint"), calibrationLayer: $("calibrationLayer"), calibrationTarget: $("calibrationTarget"), calibrationInstruction: $("calibrationInstruction"),
  calibrationProgress: $("calibrationProgress"), calibrateButton: $("calibrateButton"),
  recordingBadge: $("recordingBadge"), recordingTime: $("recordingTime"), analysisBadge: $("analysisBadge"),
  recordButton: $("recordButton"), fullscreenButton: $("fullscreenButton"),
  pipModeButton: $("pipModeButton"), hiddenModeButton: $("hiddenModeButton"),
  previewModeSwitch: $("previewModeSwitch"),
  newCaptureButton: $("newCaptureButton"), resultSummary: $("resultSummary"),
  reactionTab: $("reactionTab"), viewPanel: $("viewPanel"), reactionPanel: $("reactionPanel"),
  resultContentImage: $("resultContentImage"), resultContentVideo: $("resultContentVideo"), resultFrontVideo: $("resultFrontVideo"),
  resultYoutubeWrap: $("resultYoutubeWrap"), youtubeReactionNote: $("youtubeReactionNote"),
  viewStage: $("viewStage"), heatmapCanvas: $("heatmapCanvas"), heatmapMode: $("heatmapMode"),
  timelineCanvas: $("timelineCanvas"), timelineHelp: $("timelineHelp"), metricTracked: $("metricTracked"),
  metricPositive: $("metricPositive"), metricZone: $("metricZone"),
  reactionUnavailable: $("reactionUnavailable"), reactionAvailable: $("reactionAvailable"),
  reactionCanvas: $("reactionCanvas"), playReactionButton: $("playReactionButton"),
  pauseReactionButton: $("pauseReactionButton"),
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
let contentUrl = "";
let youtubeVideoId = "";
let contentDurationMs = 0;
let frontStream = null;
let frontRecorder = null;
let frontChunks = [];
let frontBlob = null;
let faceLandmarker = null;
let webEyeWorker = null;
let webEyeReady = false;
let webEyeBusy = false;
let webEyeLastAt = 0;
let webEyeCanvas = null;
let webEyeContext = null;
let specializedGaze = null;
let webEyeLastStepError = "";
let webEyeBackend = "";
let webEyeEyesClosed = 0;
let pendingSampleAck = null;
let pendingFitAck = null;
let calibrationFitting = false;
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
let recordingGeometry = null;
let reactionRaf = 0;
let contentResultUrl = "";
let frontResultUrl = "";
let currentCaptureId = "";
let currentCaptureCreatedAt = "";
let libraryObjectUrls = [];
let imageTimelineMs = 0;
let youtubeApiPromise = null;
let youtubeCapturePlayer = null;
let youtubeResultPlayer = null;
let youtubeResultRaf = 0;
let youtubeResultLastDrawAt = 0;

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
  const hasContent = !!selectedFile || (contentKind === "youtube" && !!youtubeVideoId);
  const ready = hasContent && els.consentAnalysis.checked;
  els.prepareButton.disabled = !ready;
  if (!hasContent) setSetupStatus("画像・動画を選ぶか、YouTube URLを入力してください");
  else if (!els.consentAnalysis.checked) setSetupStatus("端末内解析への同意を確認してください");
  else if (contentKind === "youtube") setSetupStatus("カメラ映像と解析値は端末内だけで処理し、動画再生はYouTubeへ接続します");
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
  youtubeVideoId = "";
  contentUrl = "";
  els.youtubeUrlInput.value = "";
  els.youtubeStatus.textContent = "YouTubeの共有URL、短縮URL、Shorts URLに対応します";
  els.youtubeStatus.classList.remove("error", "success");
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

function selectYouTubeUrl(rawValue, shared = false) {
  const videoId = extractYouTubeVideoId(rawValue);
  if (!videoId) {
    els.youtubeStatus.textContent = "有効なYouTube動画URLを入力してください。Netflixなど他サービスのURLには対応していません。";
    els.youtubeStatus.classList.add("error");
    els.youtubeStatus.classList.remove("success");
    return false;
  }
  releaseSelectedUrl();
  selectedFile = null;
  els.contentFileInput.value = "";
  contentBlob = null;
  contentKind = "youtube";
  youtubeVideoId = videoId;
  contentUrl = `https://www.youtube.com/watch?v=${videoId}`;
  contentName = `YouTube動画 ${videoId}`;
  contentMime = "text/uri-list";
  contentDurationMs = 0;
  els.youtubeUrlInput.value = contentUrl;
  els.selectedMediaPreview.replaceChildren();
  const preview = document.createElement("div");
  preview.className = "youtube-selected-preview";
  preview.innerHTML = "<span>▶</span><strong>YouTube</strong>";
  els.selectedMediaPreview.append(preview);
  els.selectedMediaName.textContent = contentName;
  els.selectedMediaMeta.textContent = "YouTube公式プレイヤー・再生時刻と反応を同期";
  els.youtubeStatus.textContent = shared ? "共有されたYouTube URLを読み込みました" : "YouTube URLを読み込みました";
  els.youtubeStatus.classList.remove("error");
  els.youtubeStatus.classList.add("success");
  updateReadiness();
  return true;
}

function loadYouTubeApi() {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (youtubeApiPromise) return youtubeApiPromise;
  youtubeApiPromise = new Promise((resolve, reject) => {
    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      previous?.();
      resolve(window.YT);
    };
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.async = true;
    script.onerror = () => reject(new Error("YouTubeプレイヤーを読み込めませんでした"));
    document.head.append(script);
  });
  return youtubeApiPromise;
}

function resetYoutubeTarget(wrapper, id) {
  const target = document.createElement("div");
  target.id = id;
  wrapper.replaceChildren(target);
}

async function createYoutubePlayer(targetId, videoId, onStateChange) {
  const YT = await loadYouTubeApi();
  return new Promise((resolve, reject) => {
    let ready = false;
    const player = new YT.Player(targetId, {
      width: "100%",
      height: "100%",
      videoId,
      playerVars: { playsinline: 1, rel: 0, enablejsapi: 1, origin: location.origin },
      events: {
        onReady: () => { ready = true; resolve(player); },
        onStateChange,
        onError: (event) => {
          const error = new Error(`YouTube動画を再生できません（コード ${event.data}）`);
          if (!ready) reject(error);
          else resultOrLibraryStatus(error.message);
        },
      },
    });
  });
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
  if ((!selectedFile && !youtubeVideoId) || !els.consentAnalysis.checked) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setSetupStatus("このブラウザはカメラ解析または録画に対応していません。最新版のSafari・Chrome・Edgeをお試しください。", true);
    return;
  }
  els.prepareButton.disabled = true;
  setSetupStatus("内カメと視線・表情モデルを準備しています…");
  showScreen("capture");
  try {
    await mountSelectedContent();
    frontStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 }, frameRate: { ideal: 24, max: 30 } },
      audio: false,
    });
    await attachCameraVideo(els.frontPreview, frontStream);
    await loadFaceModel();
    await loadSpecializedGazeModel();
    startAnalysisLoop();
    els.analysisBadge.querySelector("span").textContent = "専用モデルで視線・表情を端末内解析";
    calibrationModel = null;
    recordingGeometry = null;
    els.recordButton.disabled = true;
    els.captureHint.textContent = "動画の表示領域に合わせて、視線調整を行ってから記録を開始してください";
  } catch (error) {
    console.error(error);
    stopAllStreams();
    showScreen("setup");
    setSetupStatus(cameraErrorMessage(error), true);
    els.prepareButton.disabled = false;
  }
}

async function mountSelectedContent() {
  const isImage = contentKind === "image";
  const isYoutube = contentKind === "youtube";
  els.contentImage.classList.toggle("hidden", !isImage);
  els.contentVideo.classList.toggle("hidden", isImage || isYoutube);
  els.captureYoutubeWrap.classList.toggle("hidden", !isYoutube);
  els.contentTypeBadge.textContent = isImage ? "IMAGE" : isYoutube ? "YOUTUBE" : "VIDEO";
  if (isImage) {
    els.contentImage.src = selectedUrl;
    els.contentVideo.removeAttribute("src");
    els.contentVideo.load();
  } else if (!isYoutube) {
    els.contentVideo.src = selectedUrl;
    els.contentVideo.currentTime = 0;
    els.contentImage.removeAttribute("src");
  } else {
    els.contentImage.removeAttribute("src");
    els.contentVideo.removeAttribute("src");
    els.contentVideo.load();
    youtubeCapturePlayer?.destroy?.();
    resetYoutubeTarget(els.captureYoutubeWrap, "captureYoutubePlayer");
    youtubeCapturePlayer = await createYoutubePlayer("captureYoutubePlayer", youtubeVideoId, (event) => {
      if (event.data === window.YT?.PlayerState?.ENDED && recording) stopRecording();
    });
    const duration = youtubeCapturePlayer.getDuration?.() || 0;
    contentDurationMs = duration > 0 ? Math.round(duration * 1000) : contentDurationMs;
  }
}

function cameraErrorMessage(error) {
  if (error?.gazeModel) return error.message;
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
  stopSpecializedGazeModel();
  els.contentVideo.pause();
  youtubeCapturePlayer?.pauseVideo?.();
}

function loadSpecializedGazeModel() {
  if (webEyeReady && webEyeWorker) return Promise.resolve();
  stopSpecializedGazeModel();
  els.analysisBadge.querySelector("span").textContent = "専用視線モデルを読込中";
  return new Promise((resolve, reject) => {
    let worker;
    try {
      worker = new Worker(SPECIALIZED_GAZE_WORKER_URL);
    } catch (error) {
      reject(gazeModelError("専用視線モデルを開始できませんでした（このブラウザはWorkerに未対応の可能性があります）"));
      return;
    }
    webEyeWorker = worker;
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      callback(value);
    };
    const timeout = window.setTimeout(() => {
      finish(reject, gazeModelError("専用視線モデルの読み込みが時間内に終わりませんでした。通信状況を確認して、もう一度お試しください"));
    }, SPECIALIZED_GAZE_INIT_TIMEOUT_MS);
    worker.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "ready") {
        webEyeReady = true;
        webEyeBackend = message.backend || "";
        finish(resolve);
      } else if (message.type === "initError") {
        webEyeReady = false;
        finish(reject, gazeModelError(`専用視線モデルを読み込めませんでした（${message.message || "原因不明"}）`));
      } else if (message.type === "stepError") {
        webEyeBusy = false;
        webEyeLastStepError = message.message || "";
        console.warn("Specialized gaze step failed", message.message);
      } else if (message.type === "stepResult") {
        webEyeBusy = false;
        updateSpecializedGaze(message.result);
      } else if (message.type === "stepSkipped") {
        webEyeBusy = false;
      } else if (message.type === "sampleCollected") {
        pendingSampleAck?.(true);
      } else if (message.type === "sampleError") {
        pendingSampleAck?.(message.message || "unknown");
      } else if (message.type === "calibrated") {
        webEyeBusy = false;
        pendingFitAck?.(true);
      } else if (message.type === "calibrateError") {
        webEyeBusy = false;
        pendingFitAck?.(message.message || "unknown");
      } else if (message.type === "statusUpdate" && message.status === "idle") {
        webEyeBusy = false;
      }
    };
    worker.onerror = (event) => {
      webEyeBusy = false;
      webEyeReady = false;
      finish(reject, gazeModelError(event.message || "専用視線モデルを開始できませんでした"));
    };
    worker.postMessage({
      type: "init",
      payload: { modelUrl: SPECIALIZED_GAZE_MODEL_URL.href, maxPoints: CALIBRATION_POINTS.length },
    });
  });
}

function gazeModelError(message) {
  const error = new Error(message);
  error.gazeModel = true;
  return error;
}

function stopSpecializedGazeModel() {
  webEyeWorker?.terminate();
  webEyeWorker = null;
  webEyeReady = false;
  webEyeBusy = false;
  webEyeLastAt = 0;
  specializedGaze = null;
  webEyeCanvas = null;
  webEyeContext = null;
  webEyeBackend = "";
  webEyeEyesClosed = 0;
  pendingSampleAck = null;
  pendingFitAck = null;
  calibrationFitting = false;
}

function updateSpecializedGaze(result) {
  if (!result?.facialLandmarks?.length || !Array.isArray(result.normPog)) {
    specializedGaze = null;
    return;
  }
  if (result.gazeState !== "open") {
    // Blinks are normal; remember them so failures can name the real cause.
    webEyeEyesClosed += 1;
    specializedGaze = null;
    return;
  }
  const screenX = result.normPog[0] + 0.5;
  const screenY = result.normPog[1] + 0.5;
  if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) return;
  specializedGaze = {
    screenX, screenY,
    receivedAt: performance.now(),
    modelTimestamp: result.timestamp,
  };
  if (calibrationCollect && latestMetrics?.faceDetected) {
    calibrationCollect.push({
      screenX, screenY,
      yaw: latestMetrics.yaw, pitch: latestMetrics.pitch, eyeDistance: latestMetrics.eyeDistance,
    });
  }
}

function requestSpecializedGaze(now) {
  if (!webEyeReady || webEyeBusy || calibrationFitting || !webEyeWorker || els.frontPreview.readyState < 2) return;
  if (now - webEyeLastAt < SPECIALIZED_GAZE_INTERVAL_MS) return;
  webEyeLastAt = now;
  // A wider frame keeps the eye region detailed enough for the gaze model.
  const width = SPECIALIZED_GAZE_CAPTURE_WIDTH;
  const height = Math.max(2, Math.round(width * els.frontPreview.videoHeight / Math.max(els.frontPreview.videoWidth, 1)));
  if (!webEyeCanvas) {
    webEyeCanvas = document.createElement("canvas");
    webEyeContext = webEyeCanvas.getContext("2d", { willReadFrequently: true });
  }
  webEyeCanvas.width = width;
  webEyeCanvas.height = height;
  try {
    webEyeContext.drawImage(els.frontPreview, 0, 0, width, height);
    const frame = webEyeContext.getImageData(0, 0, width, height);
    webEyeBusy = true;
    webEyeWorker.postMessage({ type: "step", payload: { frame, timestamp: now } });
  } catch (error) {
    webEyeBusy = false;
    console.warn("Specialized gaze frame skipped", error);
  }
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
      requestSpecializedGaze(now);
      updateAnalysisBadge(latestMetrics);
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
  return {
    faceDetected: true, smile, furrow, browRaise, eyeOpen,
    valence: smile - furrow, yaw, pitch, eyeDistance: io,
    attention: Math.abs(yaw) < 0.35 && eyeOpen > 0.3 ? 1 : 0,
  };
}

function updateAnalysisBadge(metrics) {
  const span = els.analysisBadge.querySelector("span");
  if (!metrics?.faceDetected) span.textContent = "顔を画面側へ向けてください";
  else if (!webEyeReady) span.textContent = "専用視線モデルを読込中";
  else span.textContent = specializedGaze ? "専用モデルで視線・表情を端末内解析" : "目を確認中";
}

function projectScreenGazeToMedia(screenX, screenY, geometry = currentCaptureGeometry()) {
  if (!geometry?.media || !Number.isFinite(screenX) || !Number.isFinite(screenY)) return null;
  const captureRect = els.captureScreen.getBoundingClientRect();
  // Unclamped on purpose: clamping here would hide the shrink-to-centre bias
  // that the correction step needs to measure.
  return {
    x: (screenX * window.innerWidth - captureRect.left - geometry.media.x) / geometry.media.width,
    y: (screenY * window.innerHeight - captureRect.top - geometry.media.y) / geometry.media.height,
  };
}

function mapScreenGazeToMedia(screenX, screenY, geometry = currentCaptureGeometry(), options = {}) {
  const raw = projectScreenGazeToMedia(screenX, screenY, geometry);
  if (!raw) return null;
  const correction = "correction" in options ? options.correction : calibrationModel?.correction;
  const corrected = applyGazeCorrection(raw, correction);
  return { x: clamp(corrected.x, 0, 1), y: clamp(corrected.y, 0, 1) };
}

function applyGazeCorrection(point, correction) {
  if (!correction) return point;
  return {
    x: correction.x.scale * point.x + correction.x.offset,
    y: correction.y.scale * point.y + correction.y.offset,
  };
}

function fitGazeCorrection(samples) {
  return {
    x: fitCorrectionAxis(samples.map((s) => s.x), samples.map((s) => s.targetX)),
    y: fitCorrectionAxis(samples.map((s) => s.y), samples.map((s) => s.targetY)),
    points: samples.map((s) => ({ x: round(s.x), y: round(s.y), target_x: s.targetX, target_y: s.targetY })),
  };
}

function fitCorrectionAxis(values, targets) {
  const count = values.length;
  const meanValue = values.reduce((sum, value) => sum + value, 0) / count;
  const meanTarget = targets.reduce((sum, value) => sum + value, 0) / count;
  const variance = values.reduce((sum, value) => sum + (value - meanValue) ** 2, 0);
  const covariance = values.reduce((sum, value, index) => sum + (value - meanValue) * (targets[index] - meanTarget), 0);
  const slope = variance > 1e-4 ? covariance / variance : 0;
  // Fall back to a shift-only correction when the samples are too noisy to
  // trust a scale factor.
  if (!Number.isFinite(slope) || slope < 0.2 || slope > 8) {
    return { scale: 1, offset: meanTarget - meanValue, mode: "offset" };
  }
  return { scale: slope, offset: meanTarget - slope * meanValue, mode: "linear" };
}

function mapGaze(metrics = null, { skipPoseCheck = false } = {}) {
  if (!calibrationModel || calibrationModel.engine !== "webeyetrack" || !specializedGaze) return null;
  if (performance.now() - specializedGaze.receivedAt > SPECIALIZED_GAZE_MAX_AGE_MS) return null;
  if (!skipPoseCheck && metrics && !isCalibrationPoseStable(metrics)) return null;
  const mapped = mapScreenGazeToMedia(specializedGaze.screenX, specializedGaze.screenY);
  return mapped ? { ...mapped, calibrated: true } : null;
}

function isCalibrationPoseStable(metrics) {
  const pose = calibrationModel?.pose;
  if (!pose) return true;
  const distanceRatio = Math.abs(metrics.eyeDistance - pose.eyeDistance) / Math.max(pose.eyeDistance, 1e-6);
  if (calibrationModel?.engine === "webeyetrack") return distanceRatio <= 0.35;
  return Math.abs(metrics.yaw - pose.yaw) <= 0.16
    && Math.abs(metrics.pitch - pose.pitch) <= 0.12
    && distanceRatio <= 0.2;
}

function currentSyncMs(now = performance.now()) {
  if (contentKind === "video") return Math.round((els.contentVideo.currentTime || 0) * 1000);
  if (contentKind === "youtube") return Math.round((youtubeCapturePlayer?.getCurrentTime?.() || 0) * 1000);
  return Math.max(0, Math.round(now - recordStart));
}

function sampleMetrics(now, metrics) {
  const gaze = metrics?.faceDetected ? mapGaze(metrics) : null;
  samples.push({
    elapsed_ms: Math.max(0, Math.round(now - recordStart)),
    sync_ms: currentSyncMs(now),
    content_kind: contentKind,
    face_detected: metrics?.faceDetected ? 1 : 0,
    gaze_x: round(gaze?.x), gaze_y: round(gaze?.y), gaze_calibrated: gaze?.calibrated ? 1 : 0,
    gaze_excluded_motion: metrics?.faceDetected && specializedGaze && !gaze ? 1 : 0,
    raw_gaze_x: round(specializedGaze?.screenX), raw_gaze_y: round(specializedGaze?.screenY),
    gaze_engine: "webeyetrack",
    eye_distance: round(metrics?.eyeDistance),
    gaze_zone: gaze ? gazeZone(gaze.x, gaze.y) : "",
    attention: metrics?.attention ?? 0,
    smile: round(metrics?.smile), brow_furrow: round(metrics?.furrow),
    brow_raise: round(metrics?.browRaise), eye_open: round(metrics?.eyeOpen),
    valence: round(metrics?.valence), yaw_proxy: round(metrics?.yaw), pitch_proxy: round(metrics?.pitch),
  });
}

async function runCalibration() {
  if (!frontStream || !faceLandmarker || !webEyeReady || recording) {
    els.captureHint.textContent = "専用視線モデルを準備中です。少し待ってから視線調整を開始してください";
    return;
  }
  const geometry = currentCaptureGeometry();
  if (!geometry) {
    els.captureHint.textContent = "動画の表示サイズを確認中です。数秒待ってから視線調整を開始してください";
    return;
  }
  els.calibrateButton.disabled = true;
  els.recordButton.disabled = true;
  els.calibrationLayer.classList.remove("hidden");
  // Hide the preview switch so it cannot overlap the calibration guidance.
  els.previewModeSwitch.classList.add("hidden");
  const observations = [];
  try {
    webEyeEyesClosed = 0;
    resetSpecializedGazeCalibration();
    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
      observations.push(await collectCalibrationPoint(CALIBRATION_POINTS[i], i, CALIBRATION_POINTS.length, "動画内の点を見てください", geometry, true));
    }
    els.calibrationInstruction.textContent = "視線を学習しています…";
    els.calibrationProgress.textContent = "学習中";
    const fitted = await fitSpecializedGazeCalibration();
    if (fitted !== true) throw new Error(`視線の学習に失敗しました（${fitted}）`);
    calibrationModel = {
      engine: "webeyetrack",
      engine_version: "0.0.2",
      backend: webEyeBackend,
      calibration_points: CALIBRATION_POINTS.length,
      pose: medianPose(observations),
      viewport: currentViewport(),
      geometry,
      observations,
    };
    // Adaptation alone still leaves a systematic shrink toward the centre, so
    // measure it on a few known points and correct for it before validating.
    const correctionSamples = [];
    for (let i = 0; i < CALIBRATION_CORRECTION_POINTS.length; i++) {
      const point = CALIBRATION_CORRECTION_POINTS[i];
      const observed = await collectCalibrationPoint(point, i, CALIBRATION_CORRECTION_POINTS.length, "もう一度、点を見てください", geometry, false);
      const projected = projectScreenGazeToMedia(observed.screenX, observed.screenY, geometry);
      if (!projected) throw new Error("視線座標を動画上に変換できませんでした");
      correctionSamples.push({ ...projected, targetX: point.x, targetY: point.y });
    }
    calibrationModel.correction = fitGazeCorrection(correctionSamples);
    let qualityMessage = "視線調整が完了しました";
    try {
      const validations = [];
      for (let i = 0; i < CALIBRATION_VALIDATION_POINTS.length; i++) {
        validations.push(await collectCalibrationPoint(CALIBRATION_VALIDATION_POINTS[i], i, CALIBRATION_VALIDATION_POINTS.length, "精度を確認しています", geometry, false));
      }
      const checks = validations.map((point) => {
        const mapped = mapScreenGazeToMedia(point.screenX, point.screenY, geometry);
        if (!mapped) throw new Error("視線座標を動画上に変換できませんでした");
        const errorX = mapped.x - point.targetX;
        const errorY = mapped.y - point.targetY;
        const errorPx = Math.hypot(errorX * geometry.media.width, errorY * geometry.media.height);
        return { ...point, predictedX: round(mapped.x), predictedY: round(mapped.y), error_x: round(errorX), error_y: round(errorY), error_px: round(errorPx) };
      });
      const diagonalPx = Math.hypot(geometry.media.width, geometry.media.height);
      const meanErrorPx = checks.reduce((sum, point) => sum + point.error_px, 0) / checks.length;
      const maxErrorPx = Math.max(...checks.map((point) => point.error_px));
      const meanDiagonalRatio = meanErrorPx / diagonalPx;
      const maxDiagonalRatio = maxErrorPx / diagonalPx;
      const accepted = meanDiagonalRatio <= MAX_VALIDATION_MEAN_DIAGONAL_RATIO
        && maxDiagonalRatio <= MAX_VALIDATION_POINT_DIAGONAL_RATIO;
      calibrationModel.validation = {
        status: accepted ? "accepted" : "rejected",
        accepted,
        points: checks,
        mean_error_px: round(meanErrorPx), max_error_px: round(maxErrorPx),
        mean_diagonal_ratio: round(meanDiagonalRatio), max_diagonal_ratio: round(maxDiagonalRatio),
      };
      qualityMessage += `（確認時の平均ずれ ${Math.round(meanErrorPx)}px）`;
      if (!accepted) qualityMessage += "。精度基準を満たさないため、記録は開始できません。端末を固定して再調整してください";
    } catch (validationError) {
      console.warn("Calibration validation skipped", validationError);
      calibrationModel.validation = { status: "unavailable", accepted: false, reason: validationError.message || "unknown" };
      qualityMessage += "（精度確認を完了できなかったため、記録は開始できません）";
    }
    els.captureHint.textContent = qualityMessage;
    const accepted = calibrationModel.validation?.accepted === true;
    els.calibrateButton.innerHTML = accepted ? "<span>✓</span>調整済み" : "<span>◎</span>再調整";
    els.recordButton.disabled = !accepted;
  } catch (error) {
    console.warn(error);
    calibrationModel = null;
    els.recordButton.disabled = true;
    els.captureHint.textContent = `視線調整を完了できませんでした（${error.message || "視線を検出できませんでした"}）。端末を固定して再度お試しください`;
  } finally {
    calibrationCollect = null;
    els.calibrationLayer.classList.add("hidden");
    els.previewModeSwitch.classList.remove("hidden");
    els.calibrateButton.disabled = false;
    if (!recording) els.recordButton.disabled = calibrationModel?.validation?.accepted !== true;
  }
}

async function collectCalibrationPoint(point, index, total, instruction, geometry, trainModel) {
  els.calibrationInstruction.textContent = instruction;
  els.calibrationTarget.style.left = `${geometry.media.x + point.x * geometry.media.width}px`;
  els.calibrationTarget.style.top = `${geometry.media.y + point.y * geometry.media.height}px`;
  els.calibrationProgress.textContent = `${index + 1} / ${total}`;
  calibrationCollect = [];
  await delay(600);
  // Restart collection so samples from the previous target are never mixed in.
  calibrationCollect = [];
  webEyeLastStepError = "";
  const startedAt = performance.now();
  while (calibrationCollect.length < CALIBRATION_MIN_SAMPLES
    && performance.now() - startedAt < CALIBRATION_POINT_TIMEOUT_MS) {
    await delay(120);
  }
  if (calibrationCollect.length < CALIBRATION_MIN_SAMPLES) {
    throw new Error(calibrationPointFailureReason());
  }
  const screenX = median(calibrationCollect.map((p) => p.screenX));
  const screenY = median(calibrationCollect.map((p) => p.screenY));
  const spread = median(calibrationCollect.map((p) => Math.hypot(p.screenX - screenX, p.screenY - screenY)));
  if (spread > CALIBRATION_SPREAD_LIMIT) throw new Error("視線が大きく動いていました");
  const observation = {
    screenX, screenY, targetX: point.x, targetY: point.y,
    yaw: median(calibrationCollect.map((p) => p.yaw)),
    pitch: median(calibrationCollect.map((p) => p.pitch)),
    eyeDistance: median(calibrationCollect.map((p) => p.eyeDistance)),
  };
  if (trainModel) {
    const target = calibrationTargetToScreen(point, geometry);
    if (!webEyeWorker || !webEyeReady) throw new Error("専用視線モデルとの接続が切れました");
    const captured = await captureCalibrationSample(target);
    if (captured !== true) throw new Error(`この位置を記録できませんでした（${captured}）`);
  }
  calibrationCollect = null;
  return observation;
}

function calibrationPointFailureReason() {
  if (webEyeLastStepError) return `視線の計算に失敗しました（${webEyeLastStepError}）`;
  if (!latestMetrics?.faceDetected) return "顔を検出できませんでした";
  if (webEyeEyesClosed > 0) return "目が閉じていると判定されました。明るい場所で、目を大きく開いてお試しください";
  return "視線を十分に検出できませんでした";
}

function captureCalibrationSample(target) {
  // Snapshot only. Training is deferred until every target has been shown.
  return workerRequest(
    { type: "collectSample", payload: target },
    (finish) => { pendingSampleAck = finish; },
    () => { pendingSampleAck = null; },
    CALIBRATION_SAMPLE_TIMEOUT_MS,
  );
}

function fitSpecializedGazeCalibration() {
  // One training pass over all nine targets, so slow machines pay the cost once.
  calibrationFitting = true;
  return workerRequest(
    { type: "fitCalibration", payload: { steps: 1 } },
    (finish) => { pendingFitAck = finish; },
    () => { pendingFitAck = null; calibrationFitting = false; },
    CALIBRATION_FIT_TIMEOUT_MS,
  );
}

function resetSpecializedGazeCalibration() {
  webEyeWorker?.postMessage({ type: "resetCalibration" });
}

function workerRequest(message, register, cleanup, timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      cleanup();
      resolve(value);
    };
    const timeout = window.setTimeout(() => finish("時間内に応答がありませんでした"), timeoutMs);
    register(finish);
    webEyeWorker.postMessage(message);
  });
}

function calibrationTargetToScreen(point, geometry) {
  const captureRect = els.captureScreen.getBoundingClientRect();
  return {
    x: (captureRect.left + geometry.media.x + point.x * geometry.media.width) / window.innerWidth - 0.5,
    y: (captureRect.top + geometry.media.y + point.y * geometry.media.height) / window.innerHeight - 0.5,
  };
}

function medianPose(observations) {
  return {
    yaw: median(observations.map((point) => point.yaw)),
    pitch: median(observations.map((point) => point.pitch)),
    eyeDistance: median(observations.map((point) => point.eyeDistance)),
  };
}

function currentViewport() { return { width: window.innerWidth, height: window.innerHeight }; }

function currentCaptureGeometry() {
  const screen = els.captureScreen.getBoundingClientRect();
  const stage = els.contentStage.getBoundingClientRect();
  let mediaWidth = 16;
  let mediaHeight = 9;
  if (contentKind === "image") {
    mediaWidth = els.contentImage.naturalWidth;
    mediaHeight = els.contentImage.naturalHeight;
  } else if (contentKind === "video") {
    mediaWidth = els.contentVideo.videoWidth;
    mediaHeight = els.contentVideo.videoHeight;
  }
  if (!stage.width || !stage.height || !mediaWidth || !mediaHeight) return null;
  const scale = Math.min(stage.width / mediaWidth, stage.height / mediaHeight);
  const width = mediaWidth * scale;
  const height = mediaHeight * scale;
  return {
    coordinate_space: "capture-media-normalized",
    viewport: { width: round(screen.width), height: round(screen.height), dpr: round(window.devicePixelRatio || 1) },
    media: {
      x: round(stage.left - screen.left + (stage.width - width) / 2),
      y: round(stage.top - screen.top + (stage.height - height) / 2),
      width: round(width), height: round(height),
      aspect_ratio: round(mediaWidth / mediaHeight),
    },
  };
}

function invalidateCalibrationForViewport() {
  if (!calibrationModel?.viewport || els.captureScreen.classList.contains("hidden")) return;
  const viewport = currentViewport();
  const changed = Math.abs(viewport.width - calibrationModel.viewport.width) > 16
    || Math.abs(viewport.height - calibrationModel.viewport.height) > 16;
  if (!changed) return;
  calibrationModel = null;
  recordingGeometry = null;
  els.recordButton.disabled = true;
  els.calibrateButton.innerHTML = "<span>◎</span>視線調整";
  els.captureHint.textContent = "表示領域が変わりました。記録前に視線調整をやり直してください";
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
  if (!calibrationModel?.geometry || calibrationModel.validation?.accepted !== true) {
    els.recordButton.disabled = true;
    els.captureHint.textContent = "視線調整が必要です。動画内の点を見ながら「視線調整」を完了してください";
    return;
  }
  currentCaptureId = "";
  currentCaptureCreatedAt = "";
  frontChunks = [];
  samples = [];
  recordingGeometry = calibrationModel.geometry;
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
  } else if (contentKind === "youtube") {
    youtubeCapturePlayer?.seekTo?.(0, true);
    youtubeCapturePlayer?.playVideo?.();
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
  youtubeCapturePlayer?.pauseVideo?.();
  els.recordButton.disabled = true;
  els.captureHint.textContent = "記録を端末内でまとめています…";
  const frontType = frontRecorder?.mimeType || "video/webm";
  await stopRecorder(frontRecorder);
  frontBlob = frontChunks.length ? new Blob(frontChunks, { type: frontType }) : null;
  if (!samples.length) sampleMetrics(performance.now(), latestMetrics);
  if (contentKind === "video") contentDurationMs = Math.max(contentDurationMs, Math.round((els.contentVideo.duration || 0) * 1000));
  else if (contentKind === "youtube") contentDurationMs = Math.max(contentDurationMs, Math.round((youtubeCapturePlayer?.getDuration?.() || 0) * 1000));
  else contentDurationMs = samples.at(-1)?.elapsed_ms || 0;
  const thumbnail = await createContentThumbnail();
  stopAllStreams();
  await prepareResults();
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
  els.pipModeButton.textContent = hidden ? "小窓を再表示" : "小窓表示中";
  els.hiddenModeButton.textContent = hidden ? "非表示中" : "非表示にする";
  if (frontStream) {
    els.captureHint.textContent = hidden
      ? "内カメ映像を隠して、解析だけを続けています"
      : "内カメ映像を小窓で表示しています";
  }
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
    content_url: record.content_url || "",
    youtube_video_id: record.youtube_video_id || (record.content_kind === "youtube" ? extractYouTubeVideoId(record.content_url) : ""),
    duration_ms: record.duration_ms || normalizedSamples.at(-1)?.elapsed_ms || 0,
    samples: normalizedSamples,
    legacy_capture: legacy,
  };
}

async function createContentThumbnail() {
  if (contentKind === "youtube") return null;
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

function youtubeThumbnailUrl(videoId) {
  const safeId = String(videoId || "").trim();
  if (!/^[A-Za-z0-9_-]{6,}$/.test(safeId)) return "";
  return `https://i.ytimg.com/vi/${encodeURIComponent(safeId)}/hqdefault.jpg`;
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
    content_url: contentUrl,
    youtube_video_id: youtubeVideoId,
    front_blob: frontBlob,
    thumbnail_blob: thumbnail,
    samples,
    calibration_model: calibrationModel,
    recording_geometry: recordingGeometry,
    version: 5,
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
  const openTarget = document.createElement("button");
  openTarget.type = "button";
  openTarget.className = "library-open-target";
  openTarget.setAttribute("aria-label", `${capture.content_name}の分析を開く`);
  openTarget.addEventListener("click", () => openLibraryCapture(capture.id));
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
  } else if (capture.content_kind === "youtube") {
    const thumbnailUrl = youtubeThumbnailUrl(capture.youtube_video_id);
    if (thumbnailUrl) {
      const img = document.createElement("img");
      img.src = thumbnailUrl;
      img.alt = "YouTube動画のサムネイル";
      img.addEventListener("error", () => img.remove(), { once: true });
      thumb.append(img);
    }
  }
  const meta = document.createElement("div");
  meta.className = "library-meta";
  const title = document.createElement("strong");
  title.textContent = formatCaptureDate(capture.created_at);
  const name = document.createElement("span");
  name.textContent = capture.content_name;
  const detail = document.createElement("small");
  const legacyLabel = capture.legacy_capture ? "・旧版データ互換表示" : "";
  const kindLabel = capture.content_kind === "image" ? "画像" : capture.content_kind === "youtube" ? "YouTube" : "動画";
  detail.textContent = `${formatDuration(capture.duration_ms)}・${kindLabel}・${capture.front_blob ? "表情映像あり" : "数値解析のみ"}${legacyLabel}`;
  meta.append(title, name, detail);
  const actions = document.createElement("div");
  actions.className = "library-card-actions";
  const openButton = document.createElement("button");
  openButton.textContent = "分析を開く";
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
  openTarget.append(thumb, meta);
  card.append(openTarget, actions);
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
  contentUrl = capture.content_url;
  youtubeVideoId = capture.youtube_video_id;
  contentDurationMs = capture.content_duration_ms || capture.duration_ms;
  frontBlob = capture.front_blob || null;
  samples = capture.samples;
  calibrationModel = capture.calibration_model || null;
  currentCaptureId = capture.id;
  currentCaptureCreatedAt = capture.created_at;
  await prepareResults();
  els.saveStatus.textContent = capture.legacy_capture
    ? "旧版ライブラリの外カメ動画を、表示コンテンツとして互換表示しています。元データは変更していません。"
    : capture.content_kind === "youtube"
      ? "この端末のライブラリから開いています。解析値は端末内に保存され、再生時だけYouTubeへ接続します。"
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

async function prepareResults() {
  const isYoutube = contentKind === "youtube";
  if (!contentBlob && !isYoutube) return;
  if (contentResultUrl) URL.revokeObjectURL(contentResultUrl);
  if (frontResultUrl) URL.revokeObjectURL(frontResultUrl);
  contentResultUrl = contentBlob ? URL.createObjectURL(contentBlob) : "";
  frontResultUrl = frontBlob ? URL.createObjectURL(frontBlob) : "";
  const isImage = contentKind === "image";
  els.resultContentImage.classList.toggle("hidden", !isImage);
  els.resultContentVideo.classList.toggle("hidden", isImage || isYoutube);
  els.resultYoutubeWrap.classList.toggle("hidden", !isYoutube);
  if (isImage) {
    els.resultContentImage.src = contentResultUrl;
    els.resultContentVideo.removeAttribute("src");
    els.resultContentVideo.load();
  } else if (!isYoutube) {
    els.resultContentVideo.src = contentResultUrl;
    els.resultContentImage.removeAttribute("src");
  } else {
    els.resultContentImage.removeAttribute("src");
    els.resultContentVideo.removeAttribute("src");
    els.resultContentVideo.load();
  }
  els.resultFrontVideo.src = frontResultUrl;
  els.reactionUnavailable.classList.toggle("hidden", !!frontBlob);
  els.reactionAvailable.classList.toggle("hidden", !frontBlob);
  els.reactionTab.disabled = !frontBlob;
  els.youtubeReactionNote.classList.toggle("hidden", !isYoutube);
  els.timelineHelp.textContent = isImage ? "タイムラインをタップして経過時間を確認できます" : isYoutube ? "YouTubeの再生位置と連動します" : "動画の再生位置と連動します";
  els.downloadContentButton.textContent = isImage ? "表示画像を保存" : isYoutube ? "YouTubeで開く" : "表示動画を保存";
  imageTimelineMs = 0;
  showScreen("results");
  if (isYoutube) {
    youtubeResultPlayer?.destroy?.();
    resetYoutubeTarget(els.resultYoutubeWrap, "resultYoutubePlayer");
    youtubeResultPlayer = await createYoutubePlayer("resultYoutubePlayer", youtubeVideoId, () => {
      drawHeatmap();
      drawTimeline();
    });
    const duration = youtubeResultPlayer.getDuration?.() || 0;
    contentDurationMs = duration > 0 ? Math.round(duration * 1000) : contentDurationMs;
    startYoutubeResultLoop();
  }
  selectTab("view");
  summarizeResults();
  requestAnimationFrame(() => {
    resizeHeatmap();
    drawHeatmap();
    drawTimeline();
    if (frontBlob) drawReactionFrame();
  });
}

function startYoutubeResultLoop() {
  cancelAnimationFrame(youtubeResultRaf);
  youtubeResultLastDrawAt = 0;
  const loop = (now) => {
    if (contentKind !== "youtube" || els.resultsScreen.classList.contains("hidden")) return;
    if (now - youtubeResultLastDrawAt >= 100) {
      youtubeResultLastDrawAt = now;
      drawHeatmap();
      drawTimeline();
    }
    youtubeResultRaf = requestAnimationFrame(loop);
  };
  youtubeResultRaf = requestAnimationFrame(loop);
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
  const kindLabel = contentKind === "image" ? "画像" : contentKind === "youtube" ? "YouTube動画" : "動画";
  els.resultSummary.textContent = tracked.length
    ? `${kindLabel}と${seconds}秒間の反応から、${tracked.length}点の視線・表情データを同期しました。`
    : `${kindLabel}と反応を保存しました。この記録では視線データを十分に取得できませんでした。`;
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
  if (contentKind === "video") return els.resultContentVideo.currentTime * 1000;
  if (contentKind === "youtube") return (youtubeResultPlayer?.getCurrentTime?.() || 0) * 1000;
  return imageTimelineMs;
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
  const visible = samples.filter((sample) => sample.gaze_x !== "" && (mode === "overall" || Math.abs(sampleTime(sample) - t) <= HEATMAP_MOMENT_WINDOW_MS));
  const media = contentKind === "image" ? els.resultContentImage : els.resultContentVideo;
  const rect = contentKind === "youtube"
    ? { x: 0, y: 0, width: canvas.width, height: canvas.height }
    : displayedMediaRect(media, canvas);
  ctx.globalCompositeOperation = "lighter";
  for (const sample of visible) {
    const x = rect.x + number(sample.gaze_x) * rect.width;
    const y = rect.y + number(sample.gaze_y) * rect.height;
    const radius = Math.max(24, rect.width * (mode === "overall" ? 0.055 : 0.045));
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
  return Math.max(["video", "youtube"].includes(contentKind) ? contentDurationMs : (samples.at(-1)?.elapsed_ms || 1), 1);
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
  else if (contentKind === "youtube") youtubeResultPlayer?.seekTo?.(ms / 1000, true);
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
  if (contentKind === "youtube") {
    drawYoutubeInset(ctx, insetX, insetY, insetW, insetH, els.resultFrontVideo.currentTime * 1000);
  } else if ((contentKind === "image" && media.complete) || (contentKind === "video" && media.readyState >= 2)) {
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

function drawYoutubeInset(ctx, x, y, width, height, syncMs) {
  ctx.fillStyle = "#15171d";
  ctx.fillRect(x, y, width, height);
  ctx.fillStyle = "#ff3b30";
  roundRect(ctx, x + 16, y + 16, 48, 32, 8);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "700 18px system-ui";
  ctx.fillText("▶", x + 31, y + 39);
  ctx.font = "700 16px system-ui";
  ctx.fillText("YouTube", x + 76, y + 38);
  ctx.fillStyle = "#c8cad0";
  ctx.font = "600 13px system-ui";
  ctx.fillText(formatDuration(syncMs), x + 18, y + height - 17);
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
  const resume = els.resultFrontVideo.currentTime > 0 && els.resultFrontVideo.paused && !els.resultFrontVideo.ended;
  if (!resume) els.resultFrontVideo.currentTime = 0;
  if (contentKind === "video") {
    if (!resume) els.resultContentVideo.currentTime = 0;
    await Promise.allSettled([els.resultFrontVideo.play(), els.resultContentVideo.play()]);
  } else {
    youtubeResultPlayer?.pauseVideo?.();
    await els.resultFrontVideo.play().catch(() => {});
  }
  els.playReactionButton.textContent = "再生中…";
  els.pauseReactionButton.disabled = false;
  els.pauseReactionButton.textContent = "一時停止";
  drawReactionFrame();
  els.resultFrontVideo.addEventListener("ended", () => {
    els.resultContentVideo.pause();
    youtubeResultPlayer?.pauseVideo?.();
    els.playReactionButton.textContent = "▶ もう一度再生";
    els.pauseReactionButton.disabled = true;
  }, { once: true });
}

function pauseOrStopReaction() {
  if (!frontBlob || els.pauseReactionButton.disabled) return;
  if (!els.resultFrontVideo.paused) {
    els.resultFrontVideo.pause();
    els.resultContentVideo.pause();
    youtubeResultPlayer?.pauseVideo?.();
    cancelAnimationFrame(reactionRaf);
    drawReactionComposite(els.reactionCanvas.getContext("2d"), els.reactionCanvas, els.resultFrontVideo);
    els.playReactionButton.textContent = "▶ 再開";
    els.pauseReactionButton.textContent = "停止して最初に戻す";
    return;
  }
  els.resultFrontVideo.currentTime = 0;
  els.resultContentVideo.currentTime = 0;
  cancelAnimationFrame(reactionRaf);
  drawReactionComposite(els.reactionCanvas.getContext("2d"), els.reactionCanvas, els.resultFrontVideo);
  els.playReactionButton.textContent = "▶ リアクション映像を再生";
  els.pauseReactionButton.textContent = "一時停止";
  els.pauseReactionButton.disabled = true;
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
    schema_version: 5,
    capture_id: capture.id || "",
    created_at: capture.created_at || new Date().toISOString(),
    content: {
      kind: capture.content_kind || contentKind,
      name: capture.content_name || contentName,
      mime: capture.content_mime || contentMime,
      url: capture.content_url || contentUrl,
      youtube_video_id: capture.youtube_video_id || youtubeVideoId,
      duration_ms: capture.content_duration_ms || contentDurationMs,
    },
    synchronization: capture.content_kind === "image" ? "elapsed_ms" : capture.content_kind === "youtube" ? "youtube_playback_ms" : "content_playback_ms",
    calibration: capture.calibration_model ? "nine-point" : "uncalibrated",
    calibration_model: capture.calibration_model || null,
    recording_geometry: capture.recording_geometry || null,
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
  const files = [];
  if (capture.content_blob) {
    const contentExtension = extensionForMime(capture.content_mime || capture.content_blob.type, capture.content_kind);
    files.push(new File([capture.content_blob], `${stem}_content.${contentExtension}`, { type: capture.content_mime || capture.content_blob.type }));
  }
  if (capture.front_blob) files.push(new File([capture.front_blob], `${stem}_reaction-source.${extensionForMime(capture.front_blob.type)}`, { type: capture.front_blob.type }));
  const dataBlob = captureDataBlob(capture);
  files.push(new File([dataBlob], `${stem}_analysis.json`, { type: "application/json" }));
  return files;
}

async function shareStoredCapture(rawCapture) {
  const capture = normalizeCapture(rawCapture);
  if (!capture?.content_blob && capture?.content_kind !== "youtube") return;
  const files = captureShareFiles(capture);
  try {
    if (capture.content_kind === "youtube") {
      if (navigator.share) {
        const shareData = { title: "ViewPulseのYouTube反応記録", text: "YouTube動画と、その瞬間の反応データです。", url: capture.content_url };
        if (!navigator.canShare || navigator.canShare({ files })) shareData.files = files;
        await navigator.share(shareData);
        return;
      }
      files.forEach((file) => downloadBlob(file, file.name));
      resultOrLibraryStatus("共有機能に対応していないため、分析データを端末へ保存しました。YouTube URLはライブラリに残っています。");
      return;
    }
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
    content_url: contentUrl,
    youtube_video_id: youtubeVideoId,
    content_duration_ms: contentDurationMs,
    front_blob: frontBlob,
    samples,
    calibration_model: calibrationModel,
    recording_geometry: recordingGeometry,
    version: 5,
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
els.loadYoutubeButton.addEventListener("click", () => selectYouTubeUrl(els.youtubeUrlInput.value));
els.youtubeUrlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    selectYouTubeUrl(els.youtubeUrlInput.value);
  }
});
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
els.pauseReactionButton.addEventListener("click", pauseOrStopReaction);
els.exportReactionButton.addEventListener("click", exportReaction);
els.shareCaptureButton.addEventListener("click", () => shareStoredCapture(currentCapture()));
els.downloadContentButton.addEventListener("click", () => {
  if (contentKind === "youtube" && contentUrl) window.open(contentUrl, "_blank", "noopener");
  else if (contentBlob) downloadBlob(contentBlob, `viewpulse_content_${timestamp()}.${extensionForMime(contentMime || contentBlob.type, contentKind)}`);
});
els.downloadDataButton.addEventListener("click", () => downloadBlob(captureDataBlob(currentCapture()), `viewpulse_data_${timestamp()}.json`));
window.addEventListener("resize", () => {
  invalidateCalibrationForViewport();
  if (!els.resultsScreen.classList.contains("hidden")) { resizeHeatmap(); drawHeatmap(); drawTimeline(); }
});
document.addEventListener("fullscreenchange", () => window.setTimeout(invalidateCalibrationForViewport, 0));
window.addEventListener("pagehide", stopAllStreams);

setPreviewMode("pip");
const sharedParams = new URLSearchParams(location.search);
const sharedYoutube = sharedParams.get("source") === "share"
  ? findSharedYouTubeUrl(sharedParams.get("url"), sharedParams.get("text"))
  : null;
if (sharedYoutube) {
  selectYouTubeUrl(sharedYoutube.url, true);
  history.replaceState({}, "", location.pathname);
} else {
  updateReadiness();
}
refreshLibraryBadge();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("./service-worker.js").catch((error) => console.warn("PWA registration skipped", error));
