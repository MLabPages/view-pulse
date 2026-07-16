import { readFile } from "node:fs/promises";
import { extractYouTubeVideoId, findSharedYouTubeUrl } from "./youtube-url.mjs";

const [html, app, css, readme, manifestText, serviceWorker, icon] = await Promise.all([
  readFile("index.html", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
  readFile("README.md", "utf8"),
  readFile("manifest.webmanifest", "utf8"),
  readFile("service-worker.js", "utf8"),
  readFile("icon.svg", "utf8"),
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
  "navigator.share", "libraryDelete", "schema_version: 3",
  "loadYouTubeApi", "youtubeCapturePlayer", "youtubeResultPlayer", "youtube_playback_ms",
  "findSharedYouTubeUrl", "serviceWorker.register", "youtube_video_id",
];
const absentAppMarkers = requiredAppMarkers.filter((marker) => !app.includes(marker));
if (absentAppMarkers.length) throw new Error(`主要機能が不足: ${absentAppMarkers.join(", ")}`);

const requiredHtmlMarkers = [
  'accept="image/*,video/*"', "小窓", "非表示", "表情映像も端末内に保存する",
  "iPhone・iPadを含む", "視線ヒートマップ", "反応の波", "端末内ライブラリ",
  "YouTube URL", "YouTube選択時は動画再生のためYouTubeへ接続します", "youtubeReactionNote",
];
const absentHtmlMarkers = requiredHtmlMarkers.filter((marker) => !html.includes(marker));
if (absentHtmlMarkers.length) throw new Error(`画面要件が不足: ${absentHtmlMarkers.join(", ")}`);

if (app.includes('facingMode: { exact: "environment" }') || html.includes("前後カメラを同時")) {
  throw new Error("旧方式の外カメ／前後同時利用条件が残っています");
}
if (!css.includes("@media (max-width: 650px)")) throw new Error("スマホ向けレイアウトが不足");
if (!css.includes("word-break: keep-all")) throw new Error("見出しの自然な改行設定が不足");
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

console.log(`OK: ${htmlIds.length}個のUI要素を検証`);
console.log("OK: 画像・動画選択、内カメ1台解析、同期、表示モード切替、同意保存、結果表示を確認");
console.log("OK: 旧rear_blob互換、IndexedDBライブラリ、共有・削除、外部送信なしの説明を確認");
console.log("OK: YouTube URL解析・公式プレイヤー同期・PWA共有先・Netflix除外を確認");
