# Codex context baseline

日付: 2026-08-20

## 目的

ViewPulse のルートに Codex の入口を置き、既存の Web/PWA・視線調整・YouTube・動的 AOI の境界を次回の再調査なしに確認できるようにする。

## 確認したこと

- `README.md`、`app.js`、`manifest.webmanifest`、`local-server.mjs` を確認した。
- 端末内ファイル、YouTube、内カメラ、IndexedDB、Web Share Target が別の入力・保存境界として実装されている。
- `.webeyetrack-upstream/docs/` は upstream 資料であり、今回の `docs/` とは別である。
- 作業前の作業ツリーにコード変更はなかった。

## 変更

- ルート `AGENTS.md` と `docs/` 配下の引き継ぎ文書を追加した。
- Web アプリ本体、PWA マニフェスト、モデル、upstream 資料は変更していない。

## 検証

- `node --check app.js`
- `node --check local-server.mjs`
- `node verify.mjs`
- `git diff --check`
