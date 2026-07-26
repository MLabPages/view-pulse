import { readFile } from "node:fs/promises";
import { extractYouTubeVideoId, findSharedYouTubeUrl } from "./youtube-url.mjs";
import { applyDirectGazeMapping, evaluatePoseQuality, filterCalibrationSamples, measureAxisSeparation, median, normalizedFeature, resolveMappedGaze, selectDirectGazeMapping, signedPerpendicularFeature } from "./gaze-calibration.mjs";

const [html, app, css, readme, manifestText, serviceWorker, icon, gazeWorker, gazeModel, gazeWeights, gazeLicense] = await Promise.all([
  readFile("index.html", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
  readFile("README.md", "utf8"),
  readFile("manifest.webmanifest", "utf8"),
  readFile("service-worker.js", "utf8"),
  readFile("icon.svg", "utf8"),
  readFile("vendor/webeyetrack/webeyetrack.worker.js", "utf8"),
  readFile("web/model.json", "utf8"),
  readFile("web/group1-shard1of1.bin"),
  readFile("vendor/webeyetrack/LICENSE", "utf8"),
]);
const manifest = JSON.parse(manifestText);

const htmlIds = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
const duplicates = htmlIds.filter((id, index) => htmlIds.indexOf(id) !== index);
if (duplicates.length) throw new Error(`重複したid: ${[...new Set(duplicates)].join(", ")}`);

const referencedIds = [...app.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]);
const missing = [...new Set(referencedIds)].filter((id) => !htmlIds.includes(id));
if (missing.length) throw new Error(`HTMLに存在しないid参照: ${missing.join(", ")}`);

const requiredAppMarkers = [
  "contentFileInput", "getUserMedia", "facingMode: \"user\"", "MediaRecorder", "FaceLandmarker",
  "sync_ms", "currentSyncMs", "runCalibration", "setPreviewMode", "preview-hidden",
  "drawHeatmap", "drawTimeline", "drawReactionComposite", "exportReaction",
  "content_blob", "rear_blob", "legacy_capture", "indexedDB", "renderLibrary",
  "navigator.share", "libraryDelete", "schema_version: 6", "currentCaptureGeometry", "recording_geometry",
  "loadYouTubeApi", "youtubeCapturePlayer", "youtubeResultPlayer", "youtube_playback_ms",
  "findSharedYouTubeUrl", "serviceWorker.register", "youtube_video_id", "loadSpecializedGazeModel",
  "webeyetrack.worker.js", "gaze_engine: gazeEngine", "youtubeThumbnailUrl", "library-open-target",
  "CALIBRATION_REPEATS = 3", "CALIBRATION_MIN_SAMPLES = 3", "CALIBRATION_MAX_SAMPLES = 5",
  "randomized-three-pass-nine-point-direct-mapping", "selectDirectGazeMapping", "calibrationRequiredSamples", "gaze_quality",
  "raw_samples", "representative_points", "training_points", "waitForCalibrationTargetClick",
  "excluded_outliers", "raw_spread", "unstable", "waitForFaceAlignment", "face_center_x", "face_center_y",
  "gaze_pose_quality", "gaze_weight", "gaze_pose_deviation", "gaze_missing_reason", "model_output_stale", "renderRecordingFaceGuide", "左上の枠と点を合わせてください",
  'gazeEngine = webEyeBackend === "cpu" ? "mediapipe-iris" : "webeyetrack"', "raw_gaze_age_ms", "axis_separation", "順序が一貫しなかったため記録を開始できません", "mapped_out_of_bounds",
  "3段階の順序と確認点の精度基準を満たしたため記録できます", "vertical_error", "verticalOrderConsistent",
  "resolveMappedGaze", "gaze_at_edge", "gaze_edge_overflow",
];
const absentAppMarkers = requiredAppMarkers.filter((marker) => !app.includes(marker));
if (absentAppMarkers.length) throw new Error(`主要機能が不足: ${absentAppMarkers.join(", ")}`);
if (app.includes("fitSpecializedGazeCalibration") || app.includes("CALIBRATION_CORRECTION_POINTS") || app.includes("steps: 0")) {
  throw new Error("旧WebEyeTrack追加学習または後段5点補正が残っています");
}

const requiredHtmlMarkers = [
  'accept="image/*,video/*"', "小窓", "非表示", "表情映像も端末内に保存する",
  "iPhone・iPadを含む", "視線ヒートマップ", "反応の波", "端末内ライブラリ",
  "YouTube URL", "YouTube選択時は動画再生のためYouTubeへ接続します", "youtubeReactionNote",
  "背景の動画は調整中だけ隠れています",
  'id="calibrationTarget" class="calibration-target" type="button"',
  'id="faceAlignmentGuide"', "顔の位置・距離・向きを調整時と本測定でそろえます", 'id="recordingFaceGuide"', "正面・顔位置 OK",
];
const absentHtmlMarkers = requiredHtmlMarkers.filter((marker) => !html.includes(marker));
if (absentHtmlMarkers.length) throw new Error(`画面要件が不足: ${absentHtmlMarkers.join(", ")}`);

if (app.includes('facingMode: { exact: "environment" }') || html.includes("前後カメラを同時")) {
  throw new Error("旧方式の外カメ／前後同時利用条件が残っています");
}
if (!css.includes("@media (max-width: 650px)")) throw new Error("スマホ向けレイアウトが不足");
if (!css.includes(".calibration-layer") || !css.includes("background: #000")) throw new Error("黒背景の視線調整画面が不足");
if (!css.includes("word-break: keep-all")) throw new Error("見出しの自然な改行設定が不足");
if (!css.includes(".library-back-button") || !css.includes(".library-open-target") || !css.includes('content: "分析を見る"')) {
  throw new Error("ライブラリの戻る操作または分析を開く導線が不足");
}
if (!readme.includes("旧版") || !readme.includes("外部へ送信")) throw new Error("互換性またはプライバシー説明が不足");
if (!readme.includes("Netflix") || !readme.includes("YouTube共有から起動")) throw new Error("対応範囲の説明が不足");

const youtubeCases = [
  "https://www.youtube.com/watch?v=M7lc1UVf-VE",
  "https://youtu.be/M7lc1UVf-VE?t=3",
  "https://www.youtube.com/shorts/M7lc1UVf-VE",
  "https://www.youtube.com/embed/M7lc1UVf-VE",
];
for (const value of youtubeCases) {
  if (extractYouTubeVideoId(value) !== "M7lc1UVf-VE") throw new Error(`YouTube URL解析失敗: ${value}`);
}
if (extractYouTubeVideoId("https://www.netflix.com/watch/123")) throw new Error("YouTube以外のURLを受理しています");
const shared = findSharedYouTubeUrl("この動画を共有 https://youtu.be/M7lc1UVf-VE?t=3");
if (shared?.videoId !== "M7lc1UVf-VE") throw new Error("共有文中のYouTube URL解析に失敗");
if (manifest.share_target?.params?.url !== "url" || manifest.share_target?.action !== "./?source=share") throw new Error("PWA共有先設定が不足");
if (!manifest.icons?.some((item) => item.src === "icon.svg")) throw new Error("PWAアイコン設定が不足");
if (!serviceWorker.includes("self.clients.claim") || !icon.includes("<svg")) throw new Error("PWA起動に必要なファイルが不足");
if (!gazeWorker.includes("WebEyeTrackWorker") || !gazeModel.includes("modelTopology") || gazeWeights.length < 500_000 || !gazeLicense.includes("MIT")) {
  throw new Error("専用視線モデルまたはライセンスが不足");
}

if (median([1, 9, 3, 5]) !== 4) throw new Error("偶数サンプルの中央値計算に失敗");
if (normalizedFeature(0.75, 1, 0.5) !== 0.5 || normalizedFeature(0.75, 0.5, 1) !== 0.5) throw new Error("左右の目を同じ座標方向へ正規化できません");
const perpendicular = signedPerpendicularFeature({ x: 0.5, y: 0.1 }, { x: 0, y: 0 }, { x: 1, y: 0 });
if (Math.abs(perpendicular - 0.1) > 1e-9) throw new Error("眼軸に対する虹彩の局所縦座標計算に失敗");
const edgeGaze = resolveMappedGaze({ x: 1.21, y: 1.18 });
if (!edgeGaze || edgeGaze.x !== 1 || edgeGaze.y !== 1 || !edgeGaze.atEdge || edgeGaze.weight >= 1) {
  throw new Error("画面端へわずかにはみ出した視線を端の観測として残せません");
}
const insideGaze = resolveMappedGaze({ x: 0.62, y: 0.31 });
if (!insideGaze || insideGaze.atEdge || insideGaze.weight !== 1) throw new Error("画面内の視線を通常の観測として扱えません");
if (resolveMappedGaze({ x: 1.9, y: 0.5 }) || resolveMappedGaze({ x: 0.5, y: -0.8 })) {
  throw new Error("明らかに画面外の視線を除外できません");
}
const separatedAxes = measureAxisSeparation([0.15, 0.5, 0.85].flatMap((targetY) =>
  [0.15, 0.5, 0.85].flatMap((targetX) => [-0.002, 0, 0.002].map((noise, repeat) => ({
    targetX, targetY, repeat, screenX: targetX * 0.1 + noise, screenY: targetY * 0.1 + noise,
  })))));
if (!separatedAxes.horizontal_separated || !separatedAxes.vertical_separated) throw new Error("視線軸の分離判定に失敗");
if (!separatedAxes.vertical.monotonic || separatedAxes.vertical.point_statistics.some((point) => point.samples !== 3)) throw new Error("縦方向の3回測定統計または単調性判定に失敗");
const smallButReliableGap = measureAxisSeparation([0.15, 0.5, 0.85].flatMap((targetY, yIndex) =>
  [0.15, 0.5, 0.85].flatMap((targetX, xIndex) => [-0.0005, 0, 0.0005].map((noise, repeat) => ({
    targetX, targetY, repeat, screenX: 0.4 + xIndex * 0.01 + noise, screenY: 0.4 + yIndex * 0.01 + noise,
  })))));
if (!smallButReliableGap.vertical_separated || Math.max(...smallButReliableGap.vertical.adjacent_differences.map(Math.abs)) >= 0.025) {
  throw new Error("固定差0.025未満の安定した縦信号を利用可能と判定できません");
}
const referencePose = { faceCenterX: 0.5, faceCenterY: 0.5, yaw: 0, pitch: 0.6, eyeDistance: 0.34 };
const moderatePose = evaluatePoseQuality({ faceDetected: true, faceCenterX: 0.5, faceCenterY: 0.64, yaw: 0, pitch: 0.6, eyeDistance: 0.34 }, referencePose);
if (moderatePose.level !== "usable" || moderatePose.weight <= 0) throw new Error("通常の顔移動を利用可能として残せません");
const extremePose = evaluatePoseQuality({ faceDetected: true, faceCenterX: 0.5, faceCenterY: 0.8, yaw: 0, pitch: 0.6, eyeDistance: 0.34 }, referencePose);
if (extremePose.level !== "excluded" || extremePose.weight !== 0) throw new Error("極端な顔移動を除外できません");
const filteredSamples = filterCalibrationSamples([
  { screenX: 0.2, screenY: 0.2 },
  { screenX: 0.21, screenY: 0.19 },
  { screenX: 0.95, screenY: 0.9 },
]);
if (filteredSamples.samples.length !== 2 || filteredSamples.excludedCount !== 1) throw new Error("視線外れ値の除外に失敗");
const syntheticTargets = [0.15, 0.5, 0.85].flatMap((targetY) => [0.15, 0.5, 0.85].map((targetX) => ({
  targetX,
  targetY,
  screenX: 0.22 + 0.48 * targetX + 0.07 * targetY,
  screenY: 0.18 - 0.04 * targetX + 0.52 * targetY,
})));
const syntheticMapping = selectDirectGazeMapping(syntheticTargets);
if (!syntheticMapping || syntheticMapping.mode !== "affine2d") throw new Error("安定した一次視線変換の選択に失敗");
const syntheticErrors = syntheticTargets.map((point) => {
  const predicted = applyDirectGazeMapping(point.screenX, point.screenY, syntheticMapping);
  return Math.hypot(predicted.x - point.targetX, predicted.y - point.targetY);
});
if (Math.max(...syntheticErrors) > 0.015) throw new Error("視線座標変換の数値精度が不足");

console.log(`OK: ${htmlIds.length}個のUI要素を検証`);
console.log("OK: 画像・動画選択、内カメ1台解析、同期、表示モード切替、同意保存、結果表示を確認");
console.log("OK: 旧rear_blob互換、IndexedDBライブラリ、共有・削除、外部送信なしの説明を確認");
console.log("OK: YouTube URL解析・公式プレイヤー同期・PWA共有先・Netflix除外を確認");
console.log("OK: WebEyeTrack Worker、視線モデル重み、MITライセンスを確認");
console.log("OK: 複数回キャリブレーションの中央値・モデル選択・座標変換を数値検証");
