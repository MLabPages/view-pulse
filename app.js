import { extractYouTubeVideoId, findSharedYouTubeUrl } from "./youtube-url.mjs";
import { applyDirectGazeMapping, evaluatePoseQuality, filterCalibrationSamples, measureAxisSeparation, median, normalizedFeature, resolveMappedGaze, selectDirectGazeMapping, signedPerpendicularFeature } from "./gaze-calibration.mjs";
import { createYouTubeContentSync } from "./youtube-content-sync.mjs";
import { AOI_MIN_DWELL_MS, AOI_MISSING_GAP_MS, aoiMetricsToCsv, calculateAoiJourney, calculateAoiMetrics, librarySamplesToCsv, samplesToCsv, segmentSamples, summarizeCaptureQuality, toCsv } from "./analysis-utils.mjs";
import { DYNAMIC_AOI_DEFAULT_INTERVAL_MS, DYNAMIC_AOI_SLOW_INTERVAL_MS, assignDynamicTracks, calculateDynamicAoiMetrics, dynamicAoiAtTime } from "./dynamic-aoi-utils.mjs";

const MEDIAPIPE_MODULE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const OBJECT_MODEL = "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite";
const ANALYSIS_INTERVAL_MS = 200;
const SPECIALIZED_GAZE_INTERVAL_MS = 160;
const SPECIALIZED_GAZE_MAX_AGE_MS = 1200;
const SPECIALIZED_GAZE_INIT_TIMEOUT_MS = 45000;
const CALIBRATION_REPEATS = 3;
const CALIBRATION_MIN_SAMPLES = 3;
const CALIBRATION_MAX_SAMPLES = 5;
const CALIBRATION_SETTLE_MS = 450;
const CALIBRATION_POINT_TIMEOUT_MS = 20000;
const CALIBRATION_CLICK_TIMEOUT_MS = 15000;
const CALIBRATION_SPREAD_LIMIT = 0.18;
const SPECIALIZED_GAZE_CAPTURE_WIDTH = 640;
const SPECIALIZED_GAZE_WORKER_URL = new URL("./vendor/webeyetrack/webeyetrack.worker.js", import.meta.url);
const SPECIALIZED_GAZE_MODEL_URL = new URL("./web/model.json", import.meta.url);
const LIBRARY_DB_NAME = "viewpulse-library";
const LIBRARY_DB_VERSION = 1;
const LIBRARY_STORE = "captures";
const SCHEMA_VERSION = 7;
const EXPERIMENT_STORAGE_KEY = "viewpulse-experiment-meta";
const MAX_IMAGE_DURATION_SECONDS = 180;
const CALIBRATION_POINTS = [
  { x: 0.15, y: 0.15 }, { x: 0.5, y: 0.15 }, { x: 0.85, y: 0.15 },
  { x: 0.15, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.85, y: 0.5 },
  { x: 0.15, y: 0.85 }, { x: 0.5, y: 0.85 }, { x: 0.85, y: 0.85 },
];
const CALIBRATION_VALIDATION_POINTS = [
  { x: 0.32, y: 0.32 }, { x: 0.68, y: 0.32 },
  { x: 0.32, y: 0.68 }, { x: 0.68, y: 0.68 },
];
const HEATMAP_MOMENT_WINDOW_MS = 500;
const HEATMAP_SEGMENT_DEFAULT_SECONDS = 10;
const IMAGE_FIXATION_MS = 500;
const MAX_VALIDATION_MEAN_DIAGONAL_RATIO = 0.12;
const MAX_VALIDATION_POINT_DIAGONAL_RATIO = 0.18;

const $ = (id) => document.getElementById(id);
const els = {
  setupScreen: $("setupScreen"), captureScreen: $("captureScreen"), resultsScreen: $("resultsScreen"), libraryScreen: $("libraryScreen"),
  contentFileInput: $("contentFileInput"), selectedMediaPreview: $("selectedMediaPreview"),
  selectedMediaName: $("selectedMediaName"), selectedMediaMeta: $("selectedMediaMeta"),
  youtubeUrlInput: $("youtubeUrlInput"), youtubeStatus: $("youtubeStatus"), loadYoutubeButton: $("loadYoutubeButton"),
  consentAnalysis: $("consentAnalysis"), saveReactionVideo: $("saveReactionVideo"),
  participantIdInput: $("participantIdInput"), conditionInput: $("conditionInput"),
  imageDurationLabel: $("imageDurationLabel"), imageDurationInput: $("imageDurationInput"),
  sessionNotesInput: $("sessionNotesInput"),
  prepareButton: $("prepareButton"), setupStatus: $("setupStatus"), openLibraryButton: $("openLibraryButton"),
  closeLibraryButton: $("closeLibraryButton"), libraryCountBadge: $("libraryCountBadge"),
  libraryGrid: $("libraryGrid"), libraryEmpty: $("libraryEmpty"), storageStatus: $("storageStatus"),
  exportLibraryButton: $("exportLibraryButton"),
  contentStage: $("contentStage"), contentImage: $("contentImage"), contentVideo: $("contentVideo"),
  captureYoutubeWrap: $("captureYoutubeWrap"),
  frontPreview: $("frontPreview"), contentTypeBadge: $("contentTypeBadge"), closeCaptureButton: $("closeCaptureButton"),
  captureHint: $("captureHint"), calibrationLayer: $("calibrationLayer"), calibrationTarget: $("calibrationTarget"), calibrationInstruction: $("calibrationInstruction"),
  calibrationProgress: $("calibrationProgress"), faceAlignmentGuide: $("faceAlignmentGuide"),
  faceGuideDot: $("faceGuideDot"), faceGuideStatus: $("faceGuideStatus"), calibrateButton: $("calibrateButton"),
  recordingFaceGuide: $("recordingFaceGuide"), recordingFaceDot: $("recordingFaceDot"), recordingFaceStatus: $("recordingFaceStatus"),
  recordingBadge: $("recordingBadge"), recordingTime: $("recordingTime"), analysisBadge: $("analysisBadge"),
  recordButton: $("recordButton"), fullscreenButton: $("fullscreenButton"),
  pipModeButton: $("pipModeButton"), hiddenModeButton: $("hiddenModeButton"),
  previewModeSwitch: $("previewModeSwitch"),
  newCaptureButton: $("newCaptureButton"), resultSummary: $("resultSummary"),
  reactionTab: $("reactionTab"), viewPanel: $("viewPanel"), reactionPanel: $("reactionPanel"),
  resultContentImage: $("resultContentImage"), resultContentVideo: $("resultContentVideo"), resultFrontVideo: $("resultFrontVideo"),
  resultYoutubeWrap: $("resultYoutubeWrap"), youtubeReactionNote: $("youtubeReactionNote"),
  viewStage: $("viewStage"), heatmapCanvas: $("heatmapCanvas"), heatmapMode: $("heatmapMode"),
  analysisMode: $("analysisMode"), segmentSeconds: $("segmentSeconds"), segmentPanel: $("segmentPanel"), segmentGrid: $("segmentGrid"), segmentDetail: $("segmentDetail"),
  segmentSourcePreview: $("segmentSourcePreview"), segmentYoutubeThumbnail: $("segmentYoutubeThumbnail"),
  aoiPanel: $("aoiPanel"), aoiOverlay: $("aoiOverlay"), aoiList: $("aoiList"), aoiHelp: $("aoiHelp"), aoiJourney: $("aoiJourney"),
  dynamicAoiPanel: $("dynamicAoiPanel"), dynamicAoiOverlay: $("dynamicAoiOverlay"), dynamicAoiStatus: $("dynamicAoiStatus"),
  dynamicAoiStartButton: $("dynamicAoiStartButton"), dynamicAoiList: $("dynamicAoiList"),
  timelineCanvas: $("timelineCanvas"), timelineHelp: $("timelineHelp"), metricTracked: $("metricTracked"),
  metricPositive: $("metricPositive"), metricZone: $("metricZone"), metricCalibration: $("metricCalibration"),
  reactionUnavailable: $("reactionUnavailable"), reactionAvailable: $("reactionAvailable"),
  reactionCanvas: $("reactionCanvas"), playReactionButton: $("playReactionButton"),
  pauseReactionButton: $("pauseReactionButton"),
  exportReactionButton: $("exportReactionButton"), exportStatus: $("exportStatus"),
  downloadContentButton: $("downloadContentButton"), downloadDataButton: $("downloadDataButton"),
  downloadCsvButton: $("downloadCsvButton"),
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
let gazeEngine = "webeyetrack";
let gazeFallbackReason = "";
let webEyeEyesClosed = 0;
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
let calibrationCollectAfter = 0;
let calibrationPoseReference = null;
let recordingGeometry = null;
let reactionRaf = 0;
let contentResultUrl = "";
let frontResultUrl = "";
let currentCaptureId = "";
let currentCaptureCreatedAt = "";
let libraryObjectUrls = [];
let imageTimelineMs = 0;
let imagePresentedAt = "";
let imageAwaitingStart = false;
let heatmapSegmentSeconds = HEATMAP_SEGMENT_DEFAULT_SECONDS;
let activeHeatmapSegment = null;
let aoiRegions = [];
let dynamicAoiFrames = [];
let dynamicAoiTracks = [];
let dynamicAoiIntervalMs = DYNAMIC_AOI_DEFAULT_INTERVAL_MS;
let dynamicAoiDetector = null;
let dynamicAoiRunning = false;
let dynamicAoiCancelRequested = false;
let youtubeApiPromise = null;
let youtubeCapturePlayer = null;
let youtubeResultPlayer = null;
let youtubeResultRaf = 0;
let youtubeResultLastDrawAt = 0;
let youtubeContentSync = null;
let participantId = "";
let condition = "";
let sessionNotes = "";
let imageDurationMs = 0;
let imageStopTimer = 0;

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
  updateExperimentFieldsVisibility();
  if (!hasContent) setSetupStatus("画像・動画を選ぶか、YouTube URLを入力してください");
  else if (!els.consentAnalysis.checked) setSetupStatus("端末内解析への同意を確認してください");
  else if (contentKind === "youtube") setSetupStatus("カメラ映像と解析値は端末内だけで処理し、動画再生はYouTubeへ接続します");
  else setSetupStatus("選んだコンテンツと解析値は、この端末内だけで処理されます");
}

function updateExperimentFieldsVisibility() {
  els.imageDurationLabel.classList.toggle("hidden", contentKind !== "image");
}

function readExperimentFields() {
  participantId = els.participantIdInput.value.trim();
  condition = els.conditionInput.value.trim();
  sessionNotes = els.sessionNotesInput.value.trim();
  const seconds = Number(els.imageDurationInput.value);
  imageDurationMs = contentKind === "image" && Number.isFinite(seconds) && seconds > 0
    ? Math.round(Math.min(seconds, MAX_IMAGE_DURATION_SECONDS) * 1000)
    : 0;
  try {
    sessionStorage.setItem(EXPERIMENT_STORAGE_KEY, JSON.stringify({
      participant_id: participantId,
      condition,
      notes: sessionNotes,
      image_duration_seconds: els.imageDurationInput.value,
    }));
  } catch {}
}

function restoreExperimentFields() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(EXPERIMENT_STORAGE_KEY) || "null");
    if (saved) {
      els.participantIdInput.value = saved.participant_id || "";
      els.conditionInput.value = saved.condition || "";
      els.sessionNotesInput.value = saved.notes || "";
      els.imageDurationInput.value = saved.image_duration_seconds || "";
    }
  } catch {}
  readExperimentFields();
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
  readExperimentFields();
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
    els.analysisBadge.querySelector("span").textContent = gazeEngine === "mediapipe-iris"
      ? "CPU向け高速虹彩推定で端末内解析" : "専用モデルで視線・表情を端末内解析";
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
    els.contentImage.classList.add("content-withheld");
    imageAwaitingStart = true;
    imagePresentedAt = "";
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
  // Once this browser has proven it cannot host the Worker, keep using the
  // main-thread estimation instead of downloading the model again.
  if (gazeFallbackReason) {
    gazeEngine = "mediapipe-iris";
    return Promise.resolve();
  }
  stopSpecializedGazeModel();
  els.analysisBadge.querySelector("span").textContent = "専用視線モデルを読込中";
  return new Promise((resolve, reject) => {
    let worker;
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
    // The specialized model runs inside a Worker. Some browsers (notably
    // Chrome on iOS, whose user agent lacks the Safari "Version/" token)
    // make the bundled MediaPipe build touch `document`, which does not
    // exist in a Worker. Instead of blocking the whole capture, fall back to
    // the main-thread iris estimation that is already supported here.
    const fallback = (reason) => {
      console.warn("Specialized gaze model unavailable; using main-thread iris estimation", reason);
      webEyeWorker?.terminate();
      webEyeWorker = null;
      webEyeReady = false;
      webEyeBusy = false;
      webEyeBackend = "";
      gazeEngine = "mediapipe-iris";
      gazeFallbackReason = reason || "unknown";
      finish(resolve);
    };
    try {
      worker = new Worker(SPECIALIZED_GAZE_WORKER_URL);
    } catch (error) {
      fallback(error?.message || "worker-unsupported");
      return;
    }
    webEyeWorker = worker;
    worker.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "ready") {
        webEyeReady = true;
        webEyeBackend = message.backend || "";
        gazeEngine = webEyeBackend === "cpu" ? "mediapipe-iris" : "webeyetrack";
        gazeFallbackReason = "";
        finish(resolve);
      } else if (message.type === "initError") {
        fallback(message.message || "initError");
      } else if (message.type === "stepError") {
        webEyeBusy = false;
        webEyeLastStepError = message.message || "";
        console.warn("Specialized gaze step failed", message.message);
      } else if (message.type === "stepResult") {
        webEyeBusy = false;
        updateSpecializedGaze(message.result);
      } else if (message.type === "stepSkipped") {
        webEyeBusy = false;
      } else if (message.type === "statusUpdate" && message.status === "idle") {
        webEyeBusy = false;
      }
    };
    worker.onerror = (event) => {
      fallback(event?.message || "worker-error");
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

// Gaze estimation is ready either through the Worker-hosted specialized model
// or through the main-thread iris fallback used when the Worker cannot run.
function gazeEstimationReady() {
  return webEyeReady || gazeEngine === "mediapipe-iris";
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
  gazeEngine = "webeyetrack";
  webEyeEyesClosed = 0;
  calibrationCollectAfter = 0;
}

function updateMediaPipeGaze(now, metrics) {
  if (gazeEngine !== "mediapipe-iris" || !metrics?.faceDetected
    || !Number.isFinite(metrics.irisX) || !Number.isFinite(metrics.irisY)) return;
  specializedGaze = { screenX: metrics.irisX, screenY: metrics.irisY, receivedAt: now, modelTimestamp: now };
  if (calibrationCollect && isMetricsNearPose(metrics, calibrationPoseReference) && now >= calibrationCollectAfter) {
    calibrationCollect.push({
      screenX: metrics.irisX, screenY: metrics.irisY,
      yaw: metrics.yaw, pitch: metrics.pitch, eyeDistance: metrics.eyeDistance,
      modelTimestamp: now, receivedAt: now,
    });
  }
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
  if (calibrationCollect && latestMetrics?.faceDetected
    && isMetricsNearPose(latestMetrics, calibrationPoseReference)
    && (result.timestamp ?? 0) >= calibrationCollectAfter) {
    calibrationCollect.push({
      screenX, screenY,
      yaw: latestMetrics.yaw, pitch: latestMetrics.pitch, eyeDistance: latestMetrics.eyeDistance,
      modelTimestamp: result.timestamp, receivedAt: performance.now(),
    });
  }
}

function requestSpecializedGaze(now) {
  if (gazeEngine === "mediapipe-iris") return;
  if (!webEyeReady || webEyeBusy || !webEyeWorker || els.frontPreview.readyState < 2) return;
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
      updateMediaPipeGaze(now, latestMetrics);
      requestSpecializedGaze(now);
      updateAnalysisBadge(latestMetrics);
      if (recording) {
        const contentProgress = readYoutubeContentProgress();
        if (contentKind !== "youtube" || !contentProgress.waiting) {
          if (contentProgress.startedNow) beginActiveRecording();
          sampleMetrics(now, latestMetrics, contentProgress);
        }
      }
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
  const irisL = lm[468], irisR = lm[473];
  const eyeLInner = lm[133], eyeRInner = lm[362];
  const irisX = irisL && irisR
    ? (normalizedFeature(irisL.x, eyeL.x, eyeLInner.x) + normalizedFeature(irisR.x, eyeR.x, eyeRInner.x)) / 2
    : NaN;
  const irisY = irisL && irisR
    ? (signedPerpendicularFeature(irisL, eyeL, eyeLInner) + signedPerpendicularFeature(irisR, eyeR, eyeRInner)) / 2
    : NaN;
  return {
    faceDetected: true, smile, furrow, browRaise, eyeOpen,
    valence: smile - furrow, yaw, pitch, eyeDistance: io,
    faceCenterX: midX, faceCenterY: midY,
    irisX, irisY,
    attention: Math.abs(yaw) < 0.35 && eyeOpen > 0.3 ? 1 : 0,
  };
}

function updateAnalysisBadge(metrics) {
  const span = els.analysisBadge.querySelector("span");
  renderRecordingFaceGuide(metrics);
  if (!metrics?.faceDetected) span.textContent = "顔を画面側へ向けてください";
  else if (!gazeEstimationReady()) span.textContent = "専用視線モデルを読込中";
  else if (calibrationModel && evaluatePoseQuality(metrics, calibrationModel.pose).level === "excluded") span.textContent = "顔位置が大きく変わり視線を一時保留中";
  else span.textContent = specializedGaze
    ? (gazeEngine === "mediapipe-iris" ? "CPU向け高速虹彩推定で端末内解析" : "専用モデルで視線・表情を端末内解析")
    : "目を確認中";
}

function renderRecordingFaceGuide(metrics) {
  const pose = calibrationModel?.pose || calibrationPoseReference;
  els.recordingFaceGuide.classList.toggle("hidden", !pose || !metrics?.faceDetected);
  if (!pose || !metrics?.faceDetected) return;
  const quality = evaluatePoseQuality(metrics, pose);
  const x = clamp(50 + (pose.faceCenterX - metrics.faceCenterX) * 260, 15, 85);
  const y = clamp(50 + (metrics.faceCenterY - pose.faceCenterY) * 260, 15, 85);
  els.recordingFaceDot.style.left = `${x}%`;
  els.recordingFaceDot.style.top = `${y}%`;
  els.recordingFaceGuide.dataset.quality = quality.level;
  els.recordingFaceStatus.textContent = quality.level === "good" ? "正面・顔位置 OK" : quality.direction;
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
  const directMapping = "directMapping" in options ? options.directMapping : calibrationModel?.direct_mapping;
  if (["affine2d", "quadratic2d"].includes(directMapping?.mode)) {
    const mapped = applyDirectGazeMapping(screenX, screenY, directMapping);
    return mapped;
  }
  const raw = projectScreenGazeToMedia(screenX, screenY, geometry);
  if (!raw) return null;
  return raw;
}

function mapGaze(metrics = null, { skipPoseCheck = false } = {}) {
  if (!calibrationModel || !["webeyetrack", "mediapipe-iris"].includes(calibrationModel.engine) || !specializedGaze) return null;
  if (performance.now() - specializedGaze.receivedAt > SPECIALIZED_GAZE_MAX_AGE_MS) return null;
  const poseQuality = !skipPoseCheck && metrics ? evaluatePoseQuality(metrics, calibrationModel.pose) : { level: "good", weight: 1, severity: 0 };
  if (poseQuality.level === "excluded") return null;
  const resolved = resolveMappedGaze(mapScreenGazeToMedia(specializedGaze.screenX, specializedGaze.screenY));
  if (!resolved) return null;
  return {
    x: resolved.x, y: resolved.y, calibrated: true, poseQuality,
    atEdge: resolved.atEdge, overflow: resolved.overflow,
    weight: poseQuality.weight * resolved.weight,
  };
}

function isCalibrationPoseStable(metrics) {
  const pose = calibrationModel?.pose;
  if (!pose) return true;
  const distanceRatio = Math.abs(metrics.eyeDistance - pose.eyeDistance) / Math.max(pose.eyeDistance, 1e-6);
  if (calibrationModel?.engine === "webeyetrack") {
    const centerStable = !Number.isFinite(pose.faceCenterX)
      || (Math.abs(metrics.faceCenterX - pose.faceCenterX) <= 0.1 && Math.abs(metrics.faceCenterY - pose.faceCenterY) <= 0.1);
    return centerStable
      && Math.abs(metrics.yaw - pose.yaw) <= 0.14
      && Math.abs(metrics.pitch - pose.pitch) <= 0.1
      && distanceRatio <= 0.18;
  }
  return Math.abs(metrics.yaw - pose.yaw) <= 0.16
    && Math.abs(metrics.pitch - pose.pitch) <= 0.12
    && distanceRatio <= 0.2;
}

function currentSyncMs(now = performance.now()) {
  if (contentKind === "video") return Math.round((els.contentVideo.currentTime || 0) * 1000);
  if (contentKind === "youtube") return Math.round((youtubeCapturePlayer?.getCurrentTime?.() || 0) * 1000);
  return Math.max(0, Math.round(now - recordStart));
}

function readYoutubeContentProgress() {
  if (contentKind !== "youtube" || !youtubeContentSync) return { waiting: false, startedNow: false, validForContent: true, pauseReason: "" };
  return youtubeContentSync.observe({
    currentTime: youtubeCapturePlayer?.getCurrentTime?.(),
    playerState: youtubeCapturePlayer?.getPlayerState?.(),
    playingState: window.YT?.PlayerState?.PLAYING ?? 1,
  });
}

function beginActiveRecording() {
  recordStart = performance.now();
  frontRecorder?.start(500);
  els.recordingBadge.classList.remove("hidden");
  els.captureScreen.classList.add("is-recording");
  els.recordButton.title = "記録を終了";
  els.captureHint.textContent = "記録を開始しました";
  recordTimer = window.setInterval(updateRecordTime, 250);
  updateRecordTime();
}

function sampleMetrics(now, metrics, contentProgress = { validForContent: true, pauseReason: "" }) {
  const contentValid = contentProgress.validForContent !== false;
  const gaze = contentValid && metrics?.faceDetected ? mapGaze(metrics) : null;
  const poseQuality = metrics?.faceDetected && calibrationModel?.pose
    ? evaluatePoseQuality(metrics, calibrationModel.pose)
    : { level: "unavailable", weight: 0, severity: 0 };
  const mappedOutOfBounds = Boolean(specializedGaze)
    && !resolveMappedGaze(mapScreenGazeToMedia(specializedGaze.screenX, specializedGaze.screenY));
  const gazeMissingReason = !contentValid ? "youtube_time_not_advancing" : gaze ? ""
    : !metrics?.faceDetected ? "face_not_detected"
      : poseQuality.level === "excluded" ? "pose_excluded"
        : !specializedGaze ? "model_no_output"
          : performance.now() - specializedGaze.receivedAt > SPECIALIZED_GAZE_MAX_AGE_MS ? "model_output_stale"
            : mappedOutOfBounds ? "mapped_out_of_bounds"
            : "mapping_failed";
  samples.push({
    elapsed_ms: Math.max(0, Math.round(now - recordStart)),
    image_elapsed_ms: contentKind === "image" ? Math.max(0, Math.round(now - recordStart)) : "",
    sync_ms: currentSyncMs(now),
    content_kind: contentKind,
    gaze_valid_for_content: contentValid ? 1 : 0,
    pause_reason: contentValid ? "" : contentProgress.pauseReason || "youtube_time_not_advancing",
    face_detected: metrics?.faceDetected ? 1 : 0,
    gaze_x: round(gaze?.x), gaze_y: round(gaze?.y), gaze_calibrated: gaze?.calibrated ? 1 : 0,
    gaze_excluded_motion: gazeMissingReason === "pose_excluded" ? 1 : 0,
    gaze_missing_reason: gazeMissingReason,
    gaze_pose_quality: gaze?.poseQuality?.level || poseQuality.level,
    gaze_weight: round(gaze?.weight ?? 0),
    gaze_at_edge: gaze?.atEdge ? 1 : 0,
    gaze_edge_overflow: round(gaze?.overflow ?? 0),
    gaze_pose_deviation: round(poseQuality.severity),
    raw_gaze_x: round(specializedGaze?.screenX), raw_gaze_y: round(specializedGaze?.screenY),
    gaze_engine: gazeEngine,
    raw_gaze_timestamp_ms: round(specializedGaze?.modelTimestamp),
    raw_gaze_age_ms: round(specializedGaze ? performance.now() - specializedGaze.receivedAt : NaN),
    gaze_quality: calibrationModel?.validation?.status || "unavailable",
    eye_distance: round(metrics?.eyeDistance),
    face_center_x: round(metrics?.faceCenterX), face_center_y: round(metrics?.faceCenterY),
    gaze_zone: gaze ? gazeZone(gaze.x, gaze.y) : "",
    attention: metrics?.attention ?? 0,
    smile: round(metrics?.smile), brow_furrow: round(metrics?.furrow),
    brow_raise: round(metrics?.browRaise), eye_open: round(metrics?.eyeOpen),
    valence: round(metrics?.valence), yaw_proxy: round(metrics?.yaw), pitch_proxy: round(metrics?.pitch),
  });
}

async function runCalibration() {
  if (!frontStream || !faceLandmarker || !gazeEstimationReady() || recording) {
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
    calibrationPoseReference = await waitForFaceAlignment();
    const automatic = usesAutomaticCalibration();
    const sequence = buildCalibrationSequence();
    for (let i = 0; i < sequence.length; i++) {
      const item = sequence[i];
      observations.push(await collectCalibrationTrial(item, i, sequence.length, geometry, automatic));
    }
    const representatives = summarizeCalibrationPoints(observations);
    const axisSeparation = measureAxisSeparation(observations);
    const directMapping = selectDirectGazeMapping(representatives);
    if (!directMapping) throw new Error("9点の視線座標から変換を作成できませんでした");
    calibrationModel = {
      engine: gazeEngine,
      engine_version: gazeEngine === "webeyetrack" ? "0.0.2" : "mediapipe-face-landmarker-iris",
      backend: webEyeBackend,
      method: "randomized-three-pass-nine-point-direct-mapping",
      calibration_points: CALIBRATION_POINTS.length,
      calibration_repeats: CALIBRATION_REPEATS,
      calibration_trials: observations.length,
      samples_per_trial: { minimum: calibrationRequiredSamples(), maximum: CALIBRATION_MAX_SAMPLES },
      confirmation: automatic ? "automatic" : "click",
      pose: medianPose(observations),
      viewport: currentViewport(),
      geometry,
      observations,
      representative_points: representatives,
      axis_separation: axisSeparation,
      direct_mapping: directMapping,
    };
    calibrationModel.training_points = buildCalibrationChecks(representatives, geometry);
    let qualityMessage = "視線調整が完了しました";
    try {
      const validations = [];
      const validationSequence = shuffleCalibrationPoints(CALIBRATION_VALIDATION_POINTS);
      for (let i = 0; i < validationSequence.length; i++) {
        validations.push(await collectCalibrationTrial(validationSequence[i], i, validationSequence.length, geometry, automatic, "精度を確認しています"));
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
      const verticalOrderConsistent = axisSeparation.vertical.monotonic;
      const axesStronglySeparated = axisSeparation.horizontal_separated && axisSeparation.vertical_separated;
      // Held-out confirmation points are the primary accuracy test. Axis separation is
      // a diagnostic, because small but consistent iris motion can still map accurately.
      const validationAccuracyAccepted = meanDiagonalRatio <= MAX_VALIDATION_MEAN_DIAGONAL_RATIO
        && maxDiagonalRatio <= MAX_VALIDATION_POINT_DIAGONAL_RATIO;
      const accepted = validationAccuracyAccepted && verticalOrderConsistent;
      const verticalErrors = checks.map((point) => ({
        target_x: point.targetX, target_y: point.targetY,
        predicted_y: point.predictedY, error_y: point.error_y,
        absolute_error_px: round(Math.abs(point.error_y) * geometry.media.height),
      }));
      calibrationModel.validation = {
        status: accepted ? "accepted" : !verticalOrderConsistent ? "unusable" : "rejected",
        accepted,
        points: checks,
        mean_error_px: round(meanErrorPx), max_error_px: round(maxErrorPx),
        mean_diagonal_ratio: round(meanDiagonalRatio), max_diagonal_ratio: round(maxDiagonalRatio),
        vertical_error: {
          points: verticalErrors,
          mean_absolute_px: round(verticalErrors.reduce((sum, point) => sum + point.absolute_error_px, 0) / verticalErrors.length),
          max_absolute_px: round(Math.max(...verticalErrors.map((point) => point.absolute_error_px))),
        },
      };
      qualityMessage += `（確認時の平均ずれ ${Math.round(meanErrorPx)}px）`;
      if (accepted && !axesStronglySeparated) {
        qualityMessage += "。点間差は小さいものの、3段階の順序と確認点の精度基準を満たしたため記録できます。";
      } else if (!verticalOrderConsistent) {
        qualityMessage += "。上・中央・下の順序が一貫しなかったため記録を開始できません。正面を保って再調整してください。";
      } else if (!accepted) qualityMessage += "。低精度として記録できます。細かな注視点ではなく、大きな領域（AOI）の傾向として扱ってください。";
    } catch (validationError) {
      console.warn("Calibration validation skipped", validationError);
      calibrationModel.validation = { status: "unavailable", accepted: false, reason: validationError.message || "unknown" };
      qualityMessage += "（精度確認を完了できなかったため、記録は開始できません）";
    }
    els.captureHint.textContent = qualityMessage;
    const accepted = calibrationModel.validation?.accepted === true;
    const usable = ["accepted", "rejected"].includes(calibrationModel.validation?.status);
    els.calibrateButton.innerHTML = accepted ? "<span>✓</span>調整済み" : usable ? "<span>△</span>低精度" : "<span>◎</span>再調整";
    els.recordButton.disabled = !usable;
    els.recordButton.title = accepted ? "記録開始" : usable ? "低精度の視線データとして記録開始" : "視線調整が必要です";
  } catch (error) {
    console.warn(error);
    calibrationModel = null;
    els.recordButton.disabled = true;
    els.captureHint.textContent = `視線調整を完了できませんでした（${error.message || "視線を検出できませんでした"}）。照明・眼鏡・顔の向きでも検出が不安定になることがあります。`;
  } finally {
    calibrationCollect = null;
    calibrationCollectAfter = 0;
    calibrationPoseReference = null;
    els.faceAlignmentGuide.classList.add("hidden");
    els.calibrationTarget.classList.remove("aligning");
    els.calibrationTarget.classList.remove("ready");
    els.calibrationTarget.disabled = true;
    els.calibrationLayer.classList.add("hidden");
    els.previewModeSwitch.classList.remove("hidden");
    els.calibrateButton.disabled = false;
    if (!recording) els.recordButton.disabled = !["accepted", "rejected"].includes(calibrationModel?.validation?.status);
  }
}

function usesAutomaticCalibration() {
  const iPadDesktopMode = navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  return iPadDesktopMode || /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
}

function isMetricsNearPose(metrics, pose) {
  if (!pose || !metrics?.faceDetected) return !!metrics?.faceDetected;
  const distanceRatio = Math.abs(metrics.eyeDistance - pose.eyeDistance) / Math.max(pose.eyeDistance, 1e-6);
  return Math.abs(metrics.faceCenterX - pose.faceCenterX) <= 0.1
    && Math.abs(metrics.faceCenterY - pose.faceCenterY) <= 0.1
    && Math.abs(metrics.yaw - pose.yaw) <= 0.14
    && Math.abs(metrics.pitch - pose.pitch) <= 0.1
    && distanceRatio <= 0.18;
}

function faceGuideState(metrics) {
  if (!metrics?.faceDetected) return { ready: false, message: "顔をカメラに映してください" };
  const horizontal = metrics.faceCenterX - 0.5;
  const vertical = metrics.faceCenterY - 0.42;
  if (metrics.eyeDistance < 0.18) return { ready: false, message: "もう少しカメラへ近づいてください" };
  if (metrics.eyeDistance > 0.36) return { ready: false, message: "もう少しカメラから離れてください" };
  if (Math.abs(horizontal) > 0.13 || Math.abs(vertical) > 0.15) return { ready: false, message: "顔マークを枠の中央へ合わせてください" };
  if (Math.abs(metrics.yaw) > 0.18) return { ready: false, message: "顔を正面へ向けてください" };
  return { ready: true, message: "その位置を保ってください" };
}

function renderFaceGuide(metrics) {
  const state = faceGuideState(metrics);
  const x = metrics?.faceDetected ? clamp((1 - metrics.faceCenterX) * 100, 8, 92) : 50;
  const y = metrics?.faceDetected ? clamp(metrics.faceCenterY * 100, 8, 92) : 50;
  els.faceGuideDot.style.left = `${x}%`;
  els.faceGuideDot.style.top = `${y}%`;
  els.faceAlignmentGuide.classList.toggle("ready", state.ready);
  els.faceGuideStatus.textContent = state.message;
  return state.ready;
}

async function waitForFaceAlignment() {
  els.faceAlignmentGuide.classList.remove("hidden");
  els.calibrationTarget.classList.add("aligning");
  els.calibrationInstruction.textContent = "最初に顔の位置を合わせます";
  els.calibrationProgress.textContent = "顔位置確認";
  let stableSince = 0;
  const startedAt = performance.now();
  while (performance.now() - startedAt < 30000) {
    const ready = renderFaceGuide(latestMetrics);
    if (ready) {
      if (!stableSince) stableSince = performance.now();
      if (performance.now() - stableSince >= 800) {
        const reference = {
          yaw: latestMetrics.yaw, pitch: latestMetrics.pitch, eyeDistance: latestMetrics.eyeDistance,
          faceCenterX: latestMetrics.faceCenterX, faceCenterY: latestMetrics.faceCenterY,
        };
        els.faceAlignmentGuide.classList.add("hidden");
        els.calibrationTarget.classList.remove("aligning");
        return reference;
      }
    } else {
      stableSince = 0;
    }
    await delay(100);
  }
  if (latestMetrics?.faceDetected) {
    const reference = {
      yaw: latestMetrics.yaw, pitch: latestMetrics.pitch, eyeDistance: latestMetrics.eyeDistance,
      faceCenterX: latestMetrics.faceCenterX, faceCenterY: latestMetrics.faceCenterY,
    };
    els.faceAlignmentGuide.classList.add("hidden");
    els.calibrationTarget.classList.remove("aligning");
    return reference;
  }
  throw new Error("顔の位置を確認できませんでした。枠の中央に顔を合わせてください");
}

function shuffleCalibrationPoints(points) {
  const shuffled = points.map((point) => ({ ...point }));
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function buildCalibrationSequence(repeats = CALIBRATION_REPEATS) {
  const sequence = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    const batch = shuffleCalibrationPoints(CALIBRATION_POINTS);
    if (sequence.length && batch[0].x === sequence.at(-1).x && batch[0].y === sequence.at(-1).y) {
      [batch[0], batch[1]] = [batch[1], batch[0]];
    }
    batch.forEach((point) => sequence.push({ ...point, repeat: repeat + 1 }));
  }
  return sequence;
}

function calibrationRequiredSamples() {
  return CALIBRATION_MIN_SAMPLES;
}

async function collectCalibrationTrial(point, index, total, geometry, automatic, instruction = "") {
  els.calibrationInstruction.textContent = instruction || (automatic ? "点を見続けてください（自動取得）" : "点を見てください");
  els.calibrationTarget.style.left = `${geometry.media.x + point.x * geometry.media.width}px`;
  els.calibrationTarget.style.top = `${geometry.media.y + point.y * geometry.media.height}px`;
  els.calibrationProgress.textContent = `${index + 1} / ${total}`;
  els.calibrationTarget.classList.remove("ready");
  els.calibrationTarget.disabled = true;
  calibrationCollect = null;
  await delay(CALIBRATION_SETTLE_MS);
  calibrationCollect = [];
  calibrationCollectAfter = performance.now();
  webEyeLastStepError = "";
  webEyeEyesClosed = 0;
  const requiredSamples = calibrationRequiredSamples();
  const startedAt = performance.now();
  while (calibrationCollect.length < requiredSamples
    && performance.now() - startedAt < CALIBRATION_POINT_TIMEOUT_MS) {
    if (calibrationPoseReference && latestMetrics?.faceDetected && !isMetricsNearPose(latestMetrics, calibrationPoseReference)) {
      const poseQuality = evaluatePoseQuality(latestMetrics, calibrationPoseReference);
      els.calibrationInstruction.textContent = `${poseQuality.direction || "顔を正面へ"}（左上の枠と点を合わせてください）`;
    } else {
      els.calibrationInstruction.textContent = instruction || (automatic ? "顔を正面にして点を見続けてください（自動取得）" : "顔は正面のまま、点を見てください");
    }
    els.calibrationProgress.textContent = `${index + 1} / ${total}・視線 ${Math.min(calibrationCollect.length, requiredSamples)} / ${requiredSamples}`;
    await delay(120);
  }
  if (calibrationCollect.length < requiredSamples) {
    throw new Error(calibrationPointFailureReason());
  }
  if (automatic) {
    els.calibrationInstruction.textContent = instruction ? `${instruction}（そのまま見続けてください）` : "そのまま見続けてください（自動取得）";
    const extraUntil = performance.now() + 1000;
    while (calibrationCollect.length < CALIBRATION_MAX_SAMPLES && performance.now() < extraUntil) await delay(100);
  } else {
    els.calibrationInstruction.textContent = instruction ? `${instruction}。点を見たままクリックしてください` : "点を見たままクリックしてください";
    els.calibrationTarget.disabled = false;
    els.calibrationTarget.classList.add("ready");
    const clicked = await waitForCalibrationTargetClick(CALIBRATION_CLICK_TIMEOUT_MS);
    if (!clicked) throw new Error("注視点のクリックを確認できませんでした");
  }
  const selected = calibrationCollect.slice(-CALIBRATION_MAX_SAMPLES);
  calibrationCollect = null;
  calibrationCollectAfter = 0;
  els.calibrationTarget.disabled = true;
  els.calibrationTarget.classList.remove("ready");
  const filtered = filterCalibrationSamples(selected);
  const used = filtered.samples;
  const screenX = median(used.map((sample) => sample.screenX));
  const screenY = median(used.map((sample) => sample.screenY));
  const unstable = filtered.rawSpread > CALIBRATION_SPREAD_LIMIT
    || filtered.spread > 0.08
    || filtered.excludedCount > 0;
  return {
    screenX, screenY, targetX: point.x, targetY: point.y,
    repeat: point.repeat || 0,
    sequence: index + 1,
    sample_count: selected.length,
    used_sample_count: used.length,
    excluded_outliers: filtered.excludedCount,
    raw_spread: round(filtered.rawSpread),
    spread: round(filtered.spread),
    unstable,
    yaw: median(used.map((sample) => sample.yaw)),
    pitch: median(used.map((sample) => sample.pitch)),
    eyeDistance: median(used.map((sample) => sample.eyeDistance)),
    raw_samples: selected.map((sample) => ({
      screen_x: round(sample.screenX), screen_y: round(sample.screenY),
      yaw: round(sample.yaw), pitch: round(sample.pitch), eye_distance: round(sample.eyeDistance),
      model_timestamp: round(sample.modelTimestamp),
    })),
  };
}

function waitForCalibrationTargetClick(timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      els.calibrationTarget.removeEventListener("click", onClick);
      resolve(value);
    };
    const onClick = () => finish(true);
    const timeout = window.setTimeout(() => finish(false), timeoutMs);
    els.calibrationTarget.addEventListener("click", onClick, { once: true });
  });
}

function summarizeCalibrationPoints(observations) {
  return CALIBRATION_POINTS.map((target) => {
    const trials = observations.filter((point) => point.targetX === target.x && point.targetY === target.y);
    if (trials.length < CALIBRATION_REPEATS) throw new Error("同じ注視点を3回確認できませんでした");
    return {
      screenX: median(trials.map((point) => point.screenX)),
      screenY: median(trials.map((point) => point.screenY)),
      targetX: target.x,
      targetY: target.y,
      trial_count: trials.length,
      trial_raw_points: trials.map((point) => ({ screen_x: round(point.screenX), screen_y: round(point.screenY), sample_count: point.sample_count })),
    };
  });
}

function buildCalibrationChecks(points, geometry) {
  return points.map((point) => {
    const mapped = mapScreenGazeToMedia(point.screenX, point.screenY, geometry);
    const errorX = mapped.x - point.targetX;
    const errorY = mapped.y - point.targetY;
    return {
      raw_screen_x: round(point.screenX), raw_screen_y: round(point.screenY),
      target_x: point.targetX, target_y: point.targetY,
      predicted_x: round(mapped.x), predicted_y: round(mapped.y),
      error_x: round(errorX), error_y: round(errorY),
      error_px: round(Math.hypot(errorX * geometry.media.width, errorY * geometry.media.height)),
      trial_count: point.trial_count,
    };
  });
}

function calibrationPointFailureReason() {
  if (webEyeLastStepError) return `視線の計算に失敗しました（${webEyeLastStepError}）`;
  if (!latestMetrics?.faceDetected) return "顔を検出できませんでした";
  if (webEyeEyesClosed > 0) return "目が閉じていると判定されました。明るい場所で、目を大きく開いてお試しください";
  return "視線を十分に検出できませんでした";
}

function medianPose(observations) {
  return {
    yaw: median(observations.map((point) => point.yaw)),
    pitch: median(observations.map((point) => point.pitch)),
    eyeDistance: median(observations.map((point) => point.eyeDistance)),
    faceCenterX: calibrationPoseReference?.faceCenterX,
    faceCenterY: calibrationPoseReference?.faceCenterY,
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
  const validationStatus = calibrationModel?.validation?.status;
  if (!calibrationModel?.geometry || !["accepted", "rejected"].includes(validationStatus)) {
    els.recordButton.disabled = true;
    els.captureHint.textContent = "視線調整が必要です。動画内の点を見ながら「視線調整」を完了してください";
    return;
  }
  readExperimentFields();
  window.clearTimeout(imageStopTimer);
  imageStopTimer = 0;
  currentCaptureId = "";
  currentCaptureCreatedAt = "";
  frontChunks = [];
  samples = [];
  dynamicAoiFrames = [];
  dynamicAoiTracks = [];
  dynamicAoiIntervalMs = DYNAMIC_AOI_DEFAULT_INTERVAL_MS;
  youtubeContentSync = contentKind === "youtube" ? createYouTubeContentSync() : null;
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
  if (contentKind === "image") {
    await beginImagePresentation();
  } else if (contentKind === "video") {
    els.contentVideo.currentTime = 0;
    try { await els.contentVideo.play(); } catch (error) { console.warn("Content playback needs another tap", error); }
  } else if (contentKind === "youtube") {
    youtubeCapturePlayer?.seekTo?.(0, true);
    youtubeCapturePlayer?.playVideo?.();
  }
  if (contentKind === "image") return;
  recording = true;
  els.recordButton.classList.add("recording");
  els.recordButton.title = "記録を終了";
  els.captureScreen.classList.add("is-recording");
  if (contentKind === "youtube") {
    els.recordingBadge.classList.add("hidden");
    els.captureHint.textContent = "動画本編の開始を待っています。広告が表示された場合は，広告終了後に自動で記録を開始します。";
  } else {
    beginActiveRecording();
  }
  els.calibrateButton.disabled = true;
}

async function beginImagePresentation() {
  els.recordButton.disabled = true;
  els.captureHint.textContent = "中央の点を見てください";
  els.calibrationLayer.classList.remove("hidden");
  els.calibrationTarget.classList.remove("aligning", "ready");
  els.calibrationTarget.disabled = true;
  els.calibrationTarget.style.left = "50%";
  els.calibrationTarget.style.top = "50%";
  els.calibrationInstruction.textContent = "中央の点を見てください";
  els.calibrationProgress.textContent = "まもなく開始";
  await delay(IMAGE_FIXATION_MS);
  els.calibrationLayer.classList.add("hidden");
  els.contentImage.classList.remove("content-withheld");
  imageAwaitingStart = false;
  imagePresentedAt = new Date().toISOString();
  recording = true;
  els.recordButton.disabled = false;
  els.recordButton.classList.add("recording");
  els.recordButton.title = "記録を終了";
  els.captureScreen.classList.add("is-recording");
  beginActiveRecording();
  els.calibrateButton.disabled = true;
  if (imageDurationMs > 0) {
    els.captureHint.textContent = `画像を${Math.round(imageDurationMs / 1000)}秒表示します`;
    imageStopTimer = window.setTimeout(() => { if (recording) stopRecording(); }, imageDurationMs);
  }
}

async function stopRecording() {
  if (!recording || stopping) return;
  stopping = true;
  recording = false;
  window.clearTimeout(imageStopTimer);
  imageStopTimer = 0;
  clearInterval(recordTimer);
  youtubeContentSync = null;
  els.captureScreen.classList.remove("is-recording");
  els.recordButton.title = "記録開始";
  els.contentVideo.pause();
  youtubeCapturePlayer?.pauseVideo?.();
  els.recordButton.disabled = true;
  els.captureHint.textContent = "記録を端末内でまとめています…";
  const frontType = frontRecorder?.mimeType || "video/webm";
  await stopRecorder(frontRecorder);
  frontBlob = frontChunks.length ? new Blob(frontChunks, { type: frontType }) : null;
  if (!samples.length && contentKind !== "youtube") sampleMetrics(performance.now(), latestMetrics);
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
    gaze_valid_for_content: Number(sample.gaze_valid_for_content) === 0 ? 0 : 1,
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
    image_presented_at: record.image_presented_at || "",
    heatmap_segment_seconds: Number(record.heatmap_segment_seconds) || HEATMAP_SEGMENT_DEFAULT_SECONDS,
    aoi_regions: Array.isArray(record.aoi_regions) ? record.aoi_regions : [],
    aoi_metrics: Array.isArray(record.aoi_metrics) ? record.aoi_metrics : [],
    dynamic_aoi_interval_ms: Number(record.dynamic_aoi_interval_ms) || DYNAMIC_AOI_DEFAULT_INTERVAL_MS,
    dynamic_aoi_frames: Array.isArray(record.dynamic_aoi_frames) ? record.dynamic_aoi_frames : [],
    dynamic_aoi_tracks: Array.isArray(record.dynamic_aoi_tracks) ? record.dynamic_aoi_tracks : [],
    dynamic_aoi_metrics: Array.isArray(record.dynamic_aoi_metrics) ? record.dynamic_aoi_metrics : [],
    heatmap_segments: Array.isArray(record.heatmap_segments) ? record.heatmap_segments : [],
    samples: normalizedSamples,
    participant_id: record.participant_id || "",
    condition: record.condition || "",
    notes: record.notes || "",
    image_duration_ms: Number(record.image_duration_ms) || 0,
    quality_summary: summarizeCaptureQuality(normalizedSamples, record.calibration_model || null),
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

function youtubeThumbnailUrl(videoId, variant = "hqdefault") {
  const safeId = String(videoId || "").trim();
  if (!/^[A-Za-z0-9_-]{6,}$/.test(safeId)) return "";
  return `https://i.ytimg.com/vi/${encodeURIComponent(safeId)}/${variant}.jpg`;
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
    image_presented_at: imagePresentedAt,
    heatmap_segment_seconds: heatmapSegmentSeconds,
    aoi_regions: aoiRegions,
    aoi_metrics: calculateAllAoiMetrics(),
    dynamic_aoi_interval_ms: dynamicAoiIntervalMs,
    dynamic_aoi_frames: dynamicAoiFrames,
    dynamic_aoi_tracks: dynamicAoiTracks,
    dynamic_aoi_metrics: calculateAllDynamicAoiMetrics(),
    calibration_model: calibrationModel,
    recording_geometry: recordingGeometry,
    heatmap_segments: storedHeatmapSegments(),
    participant_id: participantId,
    condition,
    notes: sessionNotes,
    image_duration_ms: imageDurationMs,
    quality_summary: summarizeCaptureQuality(samples, calibrationModel),
    version: SCHEMA_VERSION,
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
  const quality = capture.quality_summary || summarizeCaptureQuality(capture.samples, capture.calibration_model);
  const qualityLabel = quality.gaze_quality === "accepted" ? "調整済み" : quality.gaze_quality === "rejected" ? "低精度" : "未調整";
  const participantLabel = capture.participant_id ? `・${capture.participant_id}` : "";
  const conditionLabel = capture.condition ? `・${capture.condition}` : "";
  detail.textContent = `${formatDuration(capture.duration_ms)}・${kindLabel}${participantLabel}${conditionLabel}・${qualityLabel}・${capture.front_blob ? "表情映像あり" : "数値解析のみ"}${legacyLabel}`;
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
  imagePresentedAt = capture.image_presented_at || "";
  heatmapSegmentSeconds = capture.heatmap_segment_seconds || HEATMAP_SEGMENT_DEFAULT_SECONDS;
  aoiRegions = capture.aoi_regions || [];
  dynamicAoiIntervalMs = capture.dynamic_aoi_interval_ms || DYNAMIC_AOI_DEFAULT_INTERVAL_MS;
  dynamicAoiFrames = capture.dynamic_aoi_frames || [];
  dynamicAoiTracks = capture.dynamic_aoi_tracks || [];
  calibrationModel = capture.calibration_model || null;
  participantId = capture.participant_id || "";
  condition = capture.condition || "";
  sessionNotes = capture.notes || "";
  imageDurationMs = capture.image_duration_ms || 0;
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
  heatmapSegmentSeconds = heatmapSegmentSeconds === 5 ? 5 : HEATMAP_SEGMENT_DEFAULT_SECONDS;
  activeHeatmapSegment = null;
  els.segmentSeconds.value = String(heatmapSegmentSeconds);
  els.analysisMode.value = "overall";
  els.analysisMode.querySelector('option[value="segments"]').disabled = isImage;
  els.analysisMode.querySelector('option[value="aoi"]').disabled = !isImage;
  els.analysisMode.querySelector('option[value="dynamic-aoi"]').disabled = contentKind !== "video";
  els.segmentSeconds.parentElement.classList.toggle("hidden", isImage);
  els.segmentPanel.classList.add("hidden");
  els.aoiPanel.classList.add("hidden");
  els.aoiOverlay.classList.add("hidden");
  els.dynamicAoiPanel.classList.add("hidden");
  els.dynamicAoiOverlay.classList.add("hidden");
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
    renderDynamicAoiAnalysis();
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
  const contentSamples = samples.filter((sample) => sample.gaze_valid_for_content !== 0);
  const total = contentSamples.length;
  const tracked = contentSamples.filter((sample) => sample.face_detected && sample.gaze_x !== "");
  const positive = contentSamples.filter((sample) => number(sample.smile) >= 0.35);
  const zoneCounts = {};
  tracked.forEach((sample) => { zoneCounts[sample.gaze_zone] = (zoneCounts[sample.gaze_zone] || 0) + 1; });
  const topZone = Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
  els.metricTracked.textContent = total ? `${Math.round(tracked.length / total * 100)}%` : "—";
  els.metricPositive.textContent = total ? `${Math.round(positive.length / total * 100)}%` : "—";
  els.metricZone.textContent = zoneLabel(topZone);
  const quality = summarizeCaptureQuality(samples, calibrationModel);
  els.metricCalibration.textContent = quality.gaze_quality === "accepted"
    ? (quality.validation_mean_error_px != null ? `${Math.round(quality.validation_mean_error_px)}px` : "調整済み")
    : quality.gaze_quality === "rejected" ? "低精度" : "—";
  const seconds = Math.round((contentSamples.at(-1)?.elapsed_ms || 0) / 1000);
  const kindLabel = contentKind === "image" ? "画像" : contentKind === "youtube" ? "YouTube動画" : "動画";
  const qualityStatus = quality.gaze_quality;
  const qualityPrefix = qualityStatus === "rejected" ? "低精度の視線推定です。大きな領域（AOI）の傾向として確認してください。" : "";
  const experimentPrefix = [participantId && `参加者 ${participantId}`, condition && `条件 ${condition}`].filter(Boolean).join("／");
  els.resultSummary.textContent = tracked.length
    ? `${qualityPrefix}${experimentPrefix ? `${experimentPrefix}。` : ""}${kindLabel}と${seconds}秒間の反応から、${tracked.length}点の視線・表情データを同期しました。有効視線 ${Math.round(quality.valid_gaze_ratio * 100)}%。`
    : `${experimentPrefix ? `${experimentPrefix}。` : ""}${kindLabel}と反応を保存しました。この記録では視線データを十分に取得できませんでした。`;
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
  const visible = samples.filter((sample) => sample.gaze_valid_for_content !== 0 && sample.gaze_x !== "" && (activeHeatmapSegment
    ? sampleTime(sample) >= activeHeatmapSegment.start_ms && sampleTime(sample) < activeHeatmapSegment.end_ms
    : mode === "overall" || Math.abs(sampleTime(sample) - t) <= HEATMAP_MOMENT_WINDOW_MS));
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
    const weight = Number.isFinite(Number(sample.gaze_weight)) ? clamp(Number(sample.gaze_weight), 0.35, 1) : 1;
    gradient.addColorStop(0, `rgba(255,40,20,${(mode === "overall" ? 0.16 : 0.64) * weight})`);
    gradient.addColorStop(0.34, `rgba(255,174,20,${(mode === "overall" ? 0.10 : 0.42) * weight})`);
    gradient.addColorStop(1, "rgba(255,230,40,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  ctx.globalCompositeOperation = "source-over";
}

function drawHeatmapOn(canvas, segmentRows, source, { showGrid = false } = {}) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const mediaRect = { x: 0, y: 0, width: canvas.width, height: canvas.height };
  if (showGrid) {
    ctx.strokeStyle = "rgba(255,255,255,.12)";
    ctx.lineWidth = Math.max(1, dpr);
    for (let i = 1; i < 3; i++) {
      ctx.beginPath(); ctx.moveTo(mediaRect.width * i / 3, 0); ctx.lineTo(mediaRect.width * i / 3, mediaRect.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, mediaRect.height * i / 3); ctx.lineTo(mediaRect.width, mediaRect.height * i / 3); ctx.stroke();
    }
  }
  ctx.globalCompositeOperation = "lighter";
  for (const sample of segmentRows.filter((row) => row.gaze_valid_for_content !== 0 && row.gaze_x !== "")) {
    const x = mediaRect.x + number(sample.gaze_x) * mediaRect.width;
    const y = mediaRect.y + number(sample.gaze_y) * mediaRect.height;
    const radius = Math.max(14, mediaRect.width * .09);
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, "rgba(255,40,20,.62)"); gradient.addColorStop(.38, "rgba(255,174,20,.36)"); gradient.addColorStop(1, "rgba(255,230,40,0)");
    ctx.fillStyle = gradient; ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  ctx.globalCompositeOperation = "source-over";
}

function formatRange(segment) { return `${Math.round(segment.start_ms / 1000)}～${Math.round(segment.end_ms / 1000)}秒`; }

async function renderSegmentHeatmaps() {
  if (!["video", "youtube"].includes(contentKind)) return;
  const isYoutube = contentKind === "youtube";
  const segments = segmentSamples(samples, heatmapSegmentSeconds, contentDurationMs);
  els.segmentGrid.replaceChildren();
  els.segmentDetail.textContent = activeHeatmapSegment ? `${formatRange(activeHeatmapSegment)} を大きく表示中` : isYoutube
    ? "YouTube動画では映像フレームを表示せず、時間帯別の注目分布だけを表示します。"
    : "コマを選ぶと、その時間帯だけのヒートマップを表示します";
  els.segmentSourcePreview.classList.toggle("hidden", !isYoutube);
  if (isYoutube) {
    els.segmentYoutubeThumbnail.src = youtubeThumbnailUrl(youtubeVideoId, "maxresdefault");
    els.segmentYoutubeThumbnail.onerror = () => {
      els.segmentYoutubeThumbnail.onerror = null;
      els.segmentYoutubeThumbnail.src = youtubeThumbnailUrl(youtubeVideoId, "hqdefault");
    };
  }
  const frameSource = isYoutube ? null : document.createElement("video");
  if (frameSource) {
    frameSource.src = contentResultUrl;
    frameSource.muted = true; frameSource.playsInline = true;
    await new Promise((resolve) => { frameSource.addEventListener("loadedmetadata", resolve, { once: true }); setTimeout(resolve, 1200); });
  }
  for (const segment of segments) {
    const card = document.createElement("button"); card.type = "button"; card.className = "segment-card";
    if (activeHeatmapSegment?.start_ms === segment.start_ms) card.classList.add("active");
    const frame = document.createElement("div"); frame.className = "segment-frame";
    const image = document.createElement("img"); image.alt = `${formatRange(segment)}の代表フレーム`;
    const canvas = document.createElement("canvas"); frame.append(image, canvas);
    if (isYoutube) {
      image.remove();
      const unavailable = document.createElement("small"); unavailable.className = "segment-frame-note"; unavailable.textContent = "映像フレームは表示できません"; frame.append(unavailable);
    }
    const range = document.createElement("small"); range.textContent = formatRange(segment);
    const count = document.createElement("strong"); count.textContent = `有効視線 ${segment.valid_gaze_samples}件`;
    card.append(frame, range, count);
    card.addEventListener("click", () => {
      activeHeatmapSegment = segment;
      els.heatmapMode.value = "overall";
      if (contentKind === "video") els.resultContentVideo.currentTime = (segment.start_ms + segment.end_ms) / 2000;
      else youtubeResultPlayer?.seekTo?.((segment.start_ms + segment.end_ms) / 2000, true);
      drawHeatmap(); renderSegmentHeatmaps();
    });
    els.segmentGrid.append(card);
    if (isYoutube) {
      requestAnimationFrame(() => drawHeatmapOn(canvas, segment.samples, null, { showGrid: true }));
      continue;
    }
    try {
      frameSource.currentTime = Math.min((segment.start_ms + segment.end_ms) / 2000, Math.max(0, (frameSource.duration || 0) - .05));
      await new Promise((resolve) => { frameSource.addEventListener("seeked", resolve, { once: true }); setTimeout(resolve, 800); });
      const snapshot = document.createElement("canvas"); snapshot.width = frameSource.videoWidth || 320; snapshot.height = frameSource.videoHeight || 180;
      snapshot.getContext("2d").drawImage(frameSource, 0, 0, snapshot.width, snapshot.height);
      image.src = snapshot.toDataURL("image/jpeg", .72);
    } catch { image.alt = `${formatRange(segment)}の代表フレームを作成できませんでした`; }
    requestAnimationFrame(() => drawHeatmapOn(canvas, segment.samples, image));
  }
}

function calculateAllAoiMetrics() { return aoiRegions.map((aoi) => calculateAoiMetrics(aoi, samples, { intervalMs: ANALYSIS_INTERVAL_MS, minDwellMs: AOI_MIN_DWELL_MS, missingGapMs: AOI_MISSING_GAP_MS })); }
function storedHeatmapSegments() {
  if (!["video", "youtube"].includes(contentKind)) return [];
  return segmentSamples(samples, heatmapSegmentSeconds, contentDurationMs).map(({ start_ms, end_ms, valid_gaze_samples }) => ({ start_ms, end_ms, valid_gaze_samples }));
}
async function persistResultEdits() {
  if (!currentCaptureId) return;
  try { await libraryPut(currentCapture()); } catch (error) { console.warn("Result edit save skipped", error); }
}

function renderAoiAnalysis() {
  els.aoiOverlay.replaceChildren();
  els.aoiList.replaceChildren();
  const metrics = calculateAllAoiMetrics();
  const first = metrics.filter((item) => item.first_arrival_ms != null).sort((a, b) => a.first_arrival_ms - b.first_arrival_ms)[0];
  const journey = calculateAoiJourney(aoiRegions, samples, { intervalMs: ANALYSIS_INTERVAL_MS, minDwellMs: AOI_MIN_DWELL_MS, missingGapMs: AOI_MISSING_GAP_MS });
  els.aoiHelp.textContent = aoiRegions.length ? `最初に見られたAOI: ${aoiRegions.find((aoi) => aoi.id === first?.aoi_id)?.name || "—"}` : "画像上をドラッグして領域を追加します";
  const sequenceText = journey.sequence.length ? journey.sequence.map((item) => item.aoi_name).join(" → ") : "—";
  const transitionText = journey.transitions.length ? journey.transitions.map((item) => `${item.from_name} → ${item.to_name}: ${item.count}回`).join("／") : "—";
  els.aoiJourney.innerHTML = `<strong>推定閲覧順序</strong><span>${escapeHtml(sequenceText)}</span><strong>AOI間の推定視線遷移</strong><span>${escapeHtml(transitionText)}</span><small>Seen率は複数参加者の同条件データを集計する指標です。この画面では各記録の「見た／未到達」を表示します。</small>`;
  aoiRegions.forEach((aoi) => {
    const box = document.createElement("div"); box.className = "aoi-box"; box.dataset.id = aoi.id;
    box.style.left = `${aoi.x * 100}%`; box.style.top = `${aoi.y * 100}%`; box.style.width = `${aoi.width * 100}%`; box.style.height = `${aoi.height * 100}%`;
    const label = document.createElement("span"); label.textContent = aoi.name; const resize = document.createElement("button"); resize.type = "button"; resize.className = "aoi-resize"; resize.setAttribute("aria-label", `${aoi.name}のサイズを変更`); box.append(label, resize); els.aoiOverlay.append(box);
    const metric = metrics.find((item) => item.aoi_id === aoi.id);
    const row = document.createElement("div"); row.className = "aoi-row";
    const text = document.createElement("div"); text.innerHTML = `<strong>${escapeHtml(aoi.name)}</strong><small>到達状況: ${metric?.seen ? "見た" : "未到達"}<br>推定初回到達時間: ${metric?.first_arrival_ms == null ? "—" : `${(metric.first_arrival_ms / 1000).toFixed(1)}秒`}<br>最初の推定滞在時間: ${metric?.first_dwell_ms == null ? "—" : `${(metric.first_dwell_ms / 1000).toFixed(1)}秒`}<br>推定視線滞在時間: ${((metric?.dwell_ms || 0) / 1000).toFixed(1)}秒<br>平均推定滞在時間: ${metric?.average_dwell_ms == null ? "—" : `${(metric.average_dwell_ms / 1000).toFixed(1)}秒`}<br>推定視線進入回数: ${metric?.entries || 0}回<br>推定再訪回数: ${metric?.revisits || 0}回<br>推定視線時間割合: ${Math.round((metric?.valid_time_ratio || 0) * 100)}%</small>`;
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "削除"; remove.addEventListener("click", () => { aoiRegions = aoiRegions.filter((item) => item.id !== aoi.id); renderAoiAnalysis(); persistResultEdits(); }); row.append(text, remove); els.aoiList.append(row);
  });
}

const OBJECT_LABELS_JA = {
  person: "人物", bicycle: "自転車", car: "自動車", motorcycle: "オートバイ", bus: "バス", train: "電車", truck: "トラック",
  bottle: "ボトル", cup: "カップ", fork: "フォーク", knife: "ナイフ", spoon: "スプーン", bowl: "ボウル",
  banana: "バナナ", apple: "りんご", sandwich: "サンドイッチ", orange: "オレンジ", broccoli: "ブロッコリー",
  chair: "椅子", couch: "ソファ", bed: "ベッド", "dining table": "テーブル", tv: "テレビ", laptop: "ノートPC",
  mouse: "マウス", remote: "リモコン", keyboard: "キーボード", "cell phone": "スマートフォン", book: "本", clock: "時計",
};

function calculateAllDynamicAoiMetrics() {
  return dynamicAoiTracks.filter((track) => !track.hidden)
    .map((track) => calculateDynamicAoiMetrics(track, dynamicAoiFrames, samples, { intervalMs: ANALYSIS_INTERVAL_MS }));
}

async function loadDynamicAoiDetector() {
  if (dynamicAoiDetector) return dynamicAoiDetector;
  els.dynamicAoiStatus.textContent = "物体認識モデルを端末へ読み込んでいます…";
  const { FilesetResolver, ObjectDetector } = await import(MEDIAPIPE_MODULE);
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  dynamicAoiDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: OBJECT_MODEL, delegate: "CPU" },
    runningMode: "VIDEO", scoreThreshold: 0.45, maxResults: 8,
  });
  return dynamicAoiDetector;
}

function seekVideo(video, seconds) {
  if (video.readyState >= 2 && Math.abs(video.currentTime - seconds) < 0.002) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => resolve();
    video.addEventListener("seeked", done, { once: true });
    video.currentTime = seconds;
    window.setTimeout(done, 1200);
  });
}

async function analyzeDynamicAois() {
  if (contentKind !== "video" || dynamicAoiRunning || !contentResultUrl) return;
  dynamicAoiRunning = true;
  dynamicAoiCancelRequested = false;
  els.dynamicAoiStartButton.disabled = false;
  els.dynamicAoiStartButton.textContent = "解析を中止";
  dynamicAoiFrames = [];
  dynamicAoiTracks = [];
  dynamicAoiIntervalMs = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || (navigator.hardwareConcurrency || 8) <= 4
    ? DYNAMIC_AOI_SLOW_INTERVAL_MS : DYNAMIC_AOI_DEFAULT_INTERVAL_MS;
  try {
    const detector = await loadDynamicAoiDetector();
    const source = document.createElement("video");
    source.src = contentResultUrl;
    source.muted = true;
    source.playsInline = true;
    source.preload = "auto";
    await new Promise((resolve) => {
      if (source.readyState >= 2) resolve();
      else source.addEventListener("loadeddata", resolve, { once: true });
      window.setTimeout(resolve, 2000);
    });
    if (!source.videoWidth || !source.videoHeight || source.readyState < 2) throw new Error("video_frame_unavailable");
    const durationMs = Math.max(1, Math.round((source.duration || contentDurationMs / 1000) * 1000));
    dynamicAoiIntervalMs = Math.max(dynamicAoiIntervalMs, Math.ceil(durationMs / 1200 / 500) * 500);
    let measuredTotal = 0, measuredCount = 0;
    for (let time = 0; time < durationMs; time += dynamicAoiIntervalMs) {
      if (dynamicAoiCancelRequested) throw new DOMException("cancelled", "AbortError");
      await seekVideo(source, Math.min(time / 1000, Math.max(0, (source.duration || 0) - 0.02)));
      els.dynamicAoiStatus.textContent = `物体を追跡しています… ${Math.min(100, Math.round(time / durationMs * 100))}%`;
      await new Promise(requestAnimationFrame);
      const started = performance.now();
      const result = detector.detectForVideo(source, Math.max(1, time));
      measuredTotal += performance.now() - started;
      measuredCount += 1;
      if (measuredCount === 3 && measuredTotal / measuredCount > 400) dynamicAoiIntervalMs = DYNAMIC_AOI_SLOW_INTERVAL_MS;
      const detections = (result?.detections || []).map((item) => {
        const box = item.boundingBox || {};
        const category = item.categories?.[0] || {};
        const label = category.categoryName || category.displayName || "object";
        return {
          label,
          display_name: OBJECT_LABELS_JA[label] || category.displayName || label,
          score: Math.round(Number(category.score || 0) * 1000) / 1000,
          x: clamp(number(box.originX) / Math.max(1, source.videoWidth), 0, 1),
          y: clamp(number(box.originY) / Math.max(1, source.videoHeight), 0, 1),
          width: clamp(number(box.width) / Math.max(1, source.videoWidth), 0, 1),
          height: clamp(number(box.height) / Math.max(1, source.videoHeight), 0, 1),
        };
      }).filter((item) => item.width > 0 && item.height > 0);
      dynamicAoiFrames.push({ sync_ms: time, detections });
    }
    dynamicAoiTracks = assignDynamicTracks(dynamicAoiFrames, { maxMissingMs: dynamicAoiIntervalMs * 3 });
    els.dynamicAoiStatus.textContent = dynamicAoiTracks.length
      ? `${dynamicAoiTracks.length}個の動く物体を検出しました。名称変更や不要物体の削除ができます。`
      : "この動画では対象物を十分に検出できませんでした。従来のヒートマップはそのまま利用できます。";
    renderDynamicAoiAnalysis();
    await persistResultEdits();
  } catch (error) {
    if (error?.name === "AbortError") els.dynamicAoiStatus.textContent = "物体認識を中止しました。従来のヒートマップには影響ありません。";
    else {
      console.error("Dynamic AOI analysis failed", error);
      els.dynamicAoiStatus.textContent = "この端末では物体認識を完了できませんでした。従来のヒートマップには影響ありません。";
    }
  } finally {
    dynamicAoiRunning = false;
    dynamicAoiCancelRequested = false;
    els.dynamicAoiStartButton.disabled = false;
    els.dynamicAoiStartButton.textContent = dynamicAoiFrames.length ? "物体認識を再実行" : "物体認識を開始";
  }
}

function dynamicMediaRect() {
  const stage = els.viewStage.getBoundingClientRect();
  const width = els.resultContentVideo.videoWidth || 16, height = els.resultContentVideo.videoHeight || 9;
  const scale = Math.min(stage.width / width, stage.height / height);
  const mediaWidth = width * scale, mediaHeight = height * scale;
  return { x: (stage.width - mediaWidth) / 2, y: (stage.height - mediaHeight) / 2, width: mediaWidth, height: mediaHeight };
}

function renderDynamicAoiOverlay() {
  els.dynamicAoiOverlay.replaceChildren();
  if (contentKind !== "video" || els.analysisMode.value !== "dynamic-aoi" || !dynamicAoiFrames.length) return;
  const rect = dynamicMediaRect();
  const hiddenIds = new Set(dynamicAoiTracks.filter((track) => track.hidden).map((track) => track.id));
  const tracksById = new Map(dynamicAoiTracks.map((track) => [track.id, track]));
  for (const detection of dynamicAoiAtTime(dynamicAoiFrames, resultSyncMs())) {
    if (hiddenIds.has(detection.track_id)) continue;
    const box = document.createElement("div");
    box.className = "dynamic-aoi-box";
    box.style.left = `${rect.x + detection.x * rect.width}px`;
    box.style.top = `${rect.y + detection.y * rect.height}px`;
    box.style.width = `${detection.width * rect.width}px`;
    box.style.height = `${detection.height * rect.height}px`;
    const track = tracksById.get(detection.track_id);
    box.textContent = `${track?.name || detection.display_name || detection.label} ${Math.round(detection.score * 100)}%`;
    els.dynamicAoiOverlay.append(box);
  }
}

function renderDynamicAoiAnalysis() {
  els.dynamicAoiList.replaceChildren();
  renderDynamicAoiOverlay();
  if (!dynamicAoiFrames.length) {
    els.dynamicAoiStatus.textContent = "測定後に動画フレームだけを端末内解析します。視線測定やヒートマップには影響しません。";
    return;
  }
  const metrics = calculateAllDynamicAoiMetrics();
  dynamicAoiTracks.filter((track) => !track.hidden).forEach((track) => {
    const metric = metrics.find((item) => item.track_id === track.id);
    const row = document.createElement("div"); row.className = "dynamic-aoi-row";
    const text = document.createElement("div");
    text.innerHTML = `<strong>${escapeHtml(track.name)}</strong><small>認識種類: ${escapeHtml(OBJECT_LABELS_JA[track.label] || track.label)}・検出 ${track.detections}回<br>到達状況: ${metric?.seen ? "見た" : "未到達"}<br>推定初回到達時間: ${metric?.first_arrival_ms == null ? "—" : `${(metric.first_arrival_ms / 1000).toFixed(1)}秒`}<br>推定視線滞在時間: ${((metric?.dwell_ms || 0) / 1000).toFixed(1)}秒<br>平均推定滞在時間: ${metric?.average_dwell_ms == null ? "—" : `${(metric.average_dwell_ms / 1000).toFixed(1)}秒`}<br>推定進入回数: ${metric?.entries || 0}回・推定再訪回数: ${metric?.revisits || 0}回</small>`;
    const actions = document.createElement("div"); actions.className = "dynamic-aoi-actions";
    const rename = document.createElement("button"); rename.type = "button"; rename.textContent = "名称変更";
    rename.addEventListener("click", async () => { const name = prompt("物体の名称を入力してください", track.name)?.trim(); if (name) { track.name = name; renderDynamicAoiAnalysis(); await persistResultEdits(); } });
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "除外";
    remove.addEventListener("click", async () => { track.hidden = true; renderDynamicAoiAnalysis(); await persistResultEdits(); });
    actions.append(rename, remove); row.append(text, actions); els.dynamicAoiList.append(row);
  });
}

function escapeHtml(text) { const node = document.createElement("span"); node.textContent = text; return node.innerHTML; }

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
  const contentSamples = samples.filter((sample) => sample.gaze_valid_for_content !== 0);
  drawSeries(ctx, contentSamples, duration, w, h, "valence", "#e5ff3f", (value) => 0.5 - number(value) * 0.35);
  drawSeries(ctx, contentSamples, duration, w, h, "smile", "#ff6f61", (value) => 0.92 - number(value) * 0.72);
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

function csvDownloadBlob(text) {
  return new Blob([`\uFEFF${text}`], { type: "text/csv;charset=utf-8" });
}

function analysisFileStem(capture = {}) {
  const participant = String(capture.participant_id || participantId || "anon").replace(/[^\w.-]+/g, "_").slice(0, 32) || "anon";
  return `viewpulse_${participant}_${timestamp()}`;
}

function captureAnalysisPayload(capture) {
  const rows = capture.samples || [];
  const calibration = capture.calibration_model || null;
  return {
    app: "ViewPulse",
    schema_version: SCHEMA_VERSION,
    capture_id: capture.id || "",
    created_at: capture.created_at || new Date().toISOString(),
    experiment: {
      participant_id: capture.participant_id || "",
      condition: capture.condition || "",
      notes: capture.notes || "",
      image_duration_ms: capture.image_duration_ms || 0,
    },
    content: {
      kind: capture.content_kind || contentKind,
      name: capture.content_name || contentName,
      mime: capture.content_mime || contentMime,
      url: capture.content_url || contentUrl,
      youtube_video_id: capture.youtube_video_id || youtubeVideoId,
      duration_ms: capture.content_duration_ms || contentDurationMs,
    },
    synchronization: (capture.content_kind || contentKind) === "image" ? "elapsed_ms" : (capture.content_kind || contentKind) === "youtube" ? "youtube_playback_ms" : "content_playback_ms",
    calibration: capture.calibration_model?.method || (capture.calibration_model ? "nine-point" : "uncalibrated"),
    calibration_model: calibration,
    recording_geometry: capture.recording_geometry || null,
    heatmap_segment_seconds: capture.heatmap_segment_seconds || HEATMAP_SEGMENT_DEFAULT_SECONDS,
    image_presented_at: capture.image_presented_at || "",
    aoi_regions: capture.aoi_regions || [],
    aoi_metrics: capture.aoi_metrics || [],
    dynamic_aoi_interval_ms: capture.dynamic_aoi_interval_ms || DYNAMIC_AOI_DEFAULT_INTERVAL_MS,
    dynamic_aoi_frames: capture.dynamic_aoi_frames || [],
    dynamic_aoi_tracks: capture.dynamic_aoi_tracks || [],
    dynamic_aoi_metrics: capture.dynamic_aoi_metrics || [],
    heatmap_segments: capture.heatmap_segments || storedHeatmapSegments(),
    quality: capture.quality_summary || summarizeCaptureQuality(rows, calibration),
    samples: rows,
  };
}

function captureDataBlob(capture) {
  return new Blob([JSON.stringify(captureAnalysisPayload(capture), null, 2)], { type: "application/json" });
}

function downloadCaptureCsv(capture) {
  const stem = analysisFileStem(capture);
  const rows = capture.samples || [];
  downloadBlob(csvDownloadBlob(samplesToCsv(rows)), `${stem}_samples.csv`);
  const regions = capture.aoi_regions || [];
  const aoiMetrics = capture.aoi_metrics || calculateAllAoiMetrics();
  if (regions.length) {
    window.setTimeout(() => downloadBlob(csvDownloadBlob(aoiMetricsToCsv(aoiMetrics, regions)), `${stem}_aoi.csv`), 350);
  }
  const dynamicMetrics = capture.dynamic_aoi_metrics || [];
  if (dynamicMetrics.length) {
    window.setTimeout(() => downloadBlob(csvDownloadBlob(toCsv(dynamicMetrics)), `${stem}_dynamic_aoi.csv`), 700);
  }
}

async function exportLibraryAnalysis() {
  let captures = [];
  try {
    captures = (await libraryGetAll()).map(normalizeCapture);
  } catch {
    els.storageStatus.textContent = "端末内ライブラリを開けませんでした";
    return;
  }
  if (!captures.length) {
    els.storageStatus.textContent = "保存する記録がありません";
    return;
  }
  const stem = `viewpulse_library_${timestamp()}`;
  downloadBlob(new Blob([JSON.stringify(captures.map((capture) => captureAnalysisPayload(capture)), null, 2)], { type: "application/json" }), `${stem}.json`);
  window.setTimeout(() => downloadBlob(csvDownloadBlob(librarySamplesToCsv(captures)), `${stem}_samples.csv`), 400);
  els.storageStatus.textContent = `${captures.length}件の分析データを保存しました。映像ファイルは含まれません。`;
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
    heatmap_segment_seconds: heatmapSegmentSeconds,
    image_presented_at: imagePresentedAt,
    aoi_regions: aoiRegions,
    aoi_metrics: calculateAllAoiMetrics(),
    dynamic_aoi_interval_ms: dynamicAoiIntervalMs,
    dynamic_aoi_frames: dynamicAoiFrames,
    dynamic_aoi_tracks: dynamicAoiTracks,
    dynamic_aoi_metrics: calculateAllDynamicAoiMetrics(),
    heatmap_segments: storedHeatmapSegments(),
    participant_id: participantId,
    condition,
    notes: sessionNotes,
    image_duration_ms: imageDurationMs,
    quality_summary: summarizeCaptureQuality(samples, calibrationModel),
    version: SCHEMA_VERSION,
  };
}

function timestamp() { return new Date().toISOString().replace(/[:.]/g, "-"); }
function returnToSetupForNextCapture() {
  stopAllStreams();
  cancelAnimationFrame(reactionRaf);
  cancelAnimationFrame(youtubeResultRaf);
  youtubeResultPlayer?.destroy?.();
  youtubeResultPlayer = null;
  els.resultFrontVideo.pause();
  els.resultContentVideo.pause();
  if (contentResultUrl) URL.revokeObjectURL(contentResultUrl);
  if (frontResultUrl) URL.revokeObjectURL(frontResultUrl);
  contentResultUrl = "";
  frontResultUrl = "";
  frontBlob = null;
  frontChunks = [];
  samples = [];
  aoiRegions = [];
  dynamicAoiFrames = [];
  dynamicAoiTracks = [];
  calibrationModel = null;
  recordingGeometry = null;
  currentCaptureId = "";
  currentCaptureCreatedAt = "";
  imagePresentedAt = "";
  imageAwaitingStart = false;
  activeHeatmapSegment = null;
  heatmapSegmentSeconds = HEATMAP_SEGMENT_DEFAULT_SECONDS;
  els.calibrateButton.innerHTML = "<span>◎</span>視線調整";
  els.recordButton.classList.remove("recording");
  els.recordButton.disabled = true;
  els.recordButton.title = "記録開始";
  els.captureScreen.classList.remove("is-recording");
  setPreviewMode("pip");
  if (contentKind === "youtube" && (contentUrl || youtubeVideoId)) selectYouTubeUrl(contentUrl || youtubeVideoId);
  else if (contentBlob) {
    const file = contentBlob instanceof File
      ? contentBlob
      : new File([contentBlob], contentName || "content", { type: contentMime || contentBlob.type || "application/octet-stream" });
    selectContentFile(file);
  }
  showScreen("setup");
  updateReadiness();
  if (selectedFile || youtubeVideoId) {
    setSetupStatus(participantId
      ? `同じ刺激で次の記録ができます。必要なら参加者ID（いま ${participantId}）を更新してください`
      : "同じ刺激で次の記録ができます。参加者IDを入れると後から区別しやすくなります");
  }
}
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
els.participantIdInput.addEventListener("input", readExperimentFields);
els.conditionInput.addEventListener("input", readExperimentFields);
els.sessionNotesInput.addEventListener("input", readExperimentFields);
els.imageDurationInput.addEventListener("input", readExperimentFields);
els.prepareButton.addEventListener("click", prepareCapture);
els.openLibraryButton.addEventListener("click", async () => { showScreen("library"); await renderLibrary(); });
els.closeLibraryButton.addEventListener("click", () => showScreen("setup"));
els.exportLibraryButton.addEventListener("click", exportLibraryAnalysis);
els.calibrateButton.addEventListener("click", runCalibration);
els.recordButton.addEventListener("click", () => recording ? stopRecording() : startRecording());
els.closeCaptureButton.addEventListener("click", () => {
  if (recording) {
    if (!confirm("記録を終了して結果を保存しますか？")) return;
    stopRecording();
    return;
  }
  stopAllStreams();
  showScreen("setup");
  updateReadiness();
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
els.newCaptureButton.addEventListener("click", returnToSetupForNextCapture);
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => { if (!button.disabled) selectTab(button.dataset.tab); }));
els.resultContentVideo.addEventListener("timeupdate", () => { drawHeatmap(); drawTimeline(); renderDynamicAoiOverlay(); });
els.resultContentVideo.addEventListener("loadedmetadata", () => { resizeHeatmap(); drawHeatmap(); drawTimeline(); renderDynamicAoiOverlay(); });
els.resultContentImage.addEventListener("load", () => { resizeHeatmap(); drawHeatmap(); });
els.heatmapMode.addEventListener("change", drawHeatmap);
els.analysisMode.addEventListener("change", async () => {
  const mode = els.analysisMode.value;
  activeHeatmapSegment = null;
  els.segmentPanel.classList.toggle("hidden", mode !== "segments");
  els.aoiPanel.classList.toggle("hidden", mode !== "aoi");
  els.aoiOverlay.classList.toggle("hidden", mode !== "aoi");
  els.dynamicAoiPanel.classList.toggle("hidden", mode !== "dynamic-aoi");
  els.dynamicAoiOverlay.classList.toggle("hidden", mode !== "dynamic-aoi");
  if (mode === "segments") await renderSegmentHeatmaps();
  if (mode === "aoi") renderAoiAnalysis();
  if (mode === "dynamic-aoi") renderDynamicAoiAnalysis();
  drawHeatmap();
});
els.segmentSeconds.addEventListener("change", async () => {
  heatmapSegmentSeconds = Number(els.segmentSeconds.value) === 5 ? 5 : 10;
  activeHeatmapSegment = null;
  if (els.analysisMode.value === "segments") await renderSegmentHeatmaps();
});
els.dynamicAoiStartButton.addEventListener("click", () => {
  if (dynamicAoiRunning) {
    dynamicAoiCancelRequested = true;
    els.dynamicAoiStartButton.disabled = true;
    els.dynamicAoiStatus.textContent = "物体認識を中止しています…";
  } else analyzeDynamicAois();
});
let aoiPointerState = null;
els.aoiOverlay.addEventListener("pointerdown", (event) => {
  if (contentKind !== "image") return;
  const rect = els.aoiOverlay.getBoundingClientRect();
  const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
  const box = event.target.closest(".aoi-box");
  const existing = aoiRegions.find((item) => item.id === box?.dataset.id);
  aoiPointerState = existing ? { mode: event.target.closest(".aoi-resize") ? "resize" : "move", aoi: existing, startX: x, startY: y, origin: { ...existing } } : { mode: "create", startX: x, startY: y };
  els.aoiOverlay.setPointerCapture(event.pointerId); event.preventDefault();
});
els.aoiOverlay.addEventListener("pointermove", (event) => {
  if (!aoiPointerState) return;
  const rect = els.aoiOverlay.getBoundingClientRect(); const x = clamp((event.clientX - rect.left) / rect.width, 0, 1); const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
  const state = aoiPointerState;
  if (state.mode === "create") state.preview = { x: Math.min(state.startX, x), y: Math.min(state.startY, y), width: Math.abs(x - state.startX), height: Math.abs(y - state.startY) };
  else if (state.mode === "move") { state.aoi.x = clamp(state.origin.x + x - state.startX, 0, 1 - state.origin.width); state.aoi.y = clamp(state.origin.y + y - state.startY, 0, 1 - state.origin.height); }
  else { state.aoi.width = clamp(state.origin.width + x - state.startX, .02, 1 - state.origin.x); state.aoi.height = clamp(state.origin.height + y - state.startY, .02, 1 - state.origin.y); }
  renderAoiAnalysis();
  if (state.mode === "create" && state.preview) {
    const preview = document.createElement("div"); preview.className = "aoi-box"; preview.style.left = `${state.preview.x * 100}%`; preview.style.top = `${state.preview.y * 100}%`; preview.style.width = `${state.preview.width * 100}%`; preview.style.height = `${state.preview.height * 100}%`; els.aoiOverlay.append(preview);
  }
});
els.aoiOverlay.addEventListener("pointerup", (event) => {
  const state = aoiPointerState; aoiPointerState = null;
  if (!state) return;
  if (state.mode === "create" && state.preview?.width >= .02 && state.preview?.height >= .02) {
    const name = prompt("領域名を入力してください", `要素${aoiRegions.length + 1}`)?.trim();
    if (name) aoiRegions.push({ id: crypto.randomUUID?.() || `aoi-${Date.now()}`, name, ...state.preview });
  }
  renderAoiAnalysis(); event.preventDefault();
  persistResultEdits();
});
els.timelineCanvas.addEventListener("click", seekFromTimeline);
els.playReactionButton.addEventListener("click", playReaction);
els.pauseReactionButton.addEventListener("click", pauseOrStopReaction);
els.exportReactionButton.addEventListener("click", exportReaction);
els.shareCaptureButton.addEventListener("click", () => shareStoredCapture(currentCapture()));
els.downloadContentButton.addEventListener("click", () => {
  if (contentKind === "youtube" && contentUrl) window.open(contentUrl, "_blank", "noopener");
  else if (contentBlob) downloadBlob(contentBlob, `viewpulse_content_${timestamp()}.${extensionForMime(contentMime || contentBlob.type, contentKind)}`);
});
els.downloadDataButton.addEventListener("click", () => {
  const capture = currentCapture();
  downloadBlob(captureDataBlob(capture), `${analysisFileStem(capture)}_analysis.json`);
});
els.downloadCsvButton.addEventListener("click", () => downloadCaptureCsv(currentCapture()));
window.addEventListener("resize", () => {
  invalidateCalibrationForViewport();
  if (!els.resultsScreen.classList.contains("hidden")) { resizeHeatmap(); drawHeatmap(); drawTimeline(); }
});
document.addEventListener("fullscreenchange", () => window.setTimeout(invalidateCalibrationForViewport, 0));
window.addEventListener("pagehide", stopAllStreams);
window.addEventListener("beforeunload", (event) => {
  if (!recording) return;
  event.preventDefault();
  event.returnValue = "";
});

setPreviewMode("pip");
restoreExperimentFields();
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
