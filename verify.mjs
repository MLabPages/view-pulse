import { readFile } from "node:fs/promises";

const [html, app, css] = await Promise.all([
  readFile("index.html", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
]);

const htmlIds = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
const duplicates = htmlIds.filter((id, index) => htmlIds.indexOf(id) !== index);
if (duplicates.length) throw new Error(`重複したid: ${[...new Set(duplicates)].join(", ")}`);

const referencedIds = [...app.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]);
const missing = [...new Set(referencedIds)].filter((id) => !htmlIds.includes(id));
if (missing.length) throw new Error(`HTMLに存在しないid参照: ${missing.join(", ")}`);

const requiredMarkers = [
  "getUserMedia", "MediaRecorder", "FaceLandmarker", "runCalibration",
  "drawHeatmap", "drawReactionComposite", "exportReaction",
];
const absentMarkers = requiredMarkers.filter((marker) => !app.includes(marker));
if (absentMarkers.length) throw new Error(`主要機能が不足: ${absentMarkers.join(", ")}`);
if (!css.includes("@media (max-width: 650px)")) throw new Error("スマホ向けレイアウトが不足");

console.log(`OK: ${htmlIds.length}個のUI要素を検証`);
console.log("OK: 前後カメラ・表情解析・視線調整・ヒートマップ・リアクション映像を確認");
