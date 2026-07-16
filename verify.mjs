import { readFile } from "node:fs/promises";

const [html, app, css, readme] = await Promise.all([
  readFile("index.html", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
  readFile("README.md", "utf8"),
]);

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
  "navigator.share", "libraryDelete", "schema_version: 2",
];
const absentAppMarkers = requiredAppMarkers.filter((marker) => !app.includes(marker));
if (absentAppMarkers.length) throw new Error(`主要機能が不足: ${absentAppMarkers.join(", ")}`);

const requiredHtmlMarkers = [
  'accept="image/*,video/*"', "小窓", "非表示", "表情映像も端末内に保存する",
  "iPhone・iPadを含む", "視線ヒートマップ", "反応の波", "端末内ライブラリ",
];
const absentHtmlMarkers = requiredHtmlMarkers.filter((marker) => !html.includes(marker));
if (absentHtmlMarkers.length) throw new Error(`画面要件が不足: ${absentHtmlMarkers.join(", ")}`);

if (app.includes('facingMode: { exact: "environment" }') || html.includes("前後カメラを同時")) {
  throw new Error("旧方式の外カメ／前後同時利用条件が残っています");
}
if (!css.includes("@media (max-width: 650px)")) throw new Error("スマホ向けレイアウトが不足");
if (!css.includes("word-break: keep-all")) throw new Error("見出しの自然な改行設定が不足");
if (!readme.includes("旧版") || !readme.includes("外部へ送信")) throw new Error("互換性またはプライバシー説明が不足");

console.log(`OK: ${htmlIds.length}個のUI要素を検証`);
console.log("OK: 画像・動画選択、内カメ1台解析、同期、表示モード切替、同意保存、結果表示を確認");
console.log("OK: 旧rear_blob互換、IndexedDBライブラリ、共有・削除、外部送信なしの説明を確認");
