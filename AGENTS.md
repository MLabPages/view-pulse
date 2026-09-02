# ViewPulse project guide

## 入口

ViewPulse は、端末内の画像・動画・YouTube と、内カメラから推定した視線・表情を結び付ける Web/PWA アプリです。利用方法と検証コマンドは `README.md` を参照してください。

作業開始時は、必要な範囲で次を確認します。

- `docs/current-state.md`：現在のモード・保存・対応境界
- `docs/decisions.md`：プライバシー、視線調整、YouTube、動的 AOI の判断
- `docs/dev/logs/`：関係する最近の作業記録
- `.webeyetrack-upstream/docs/`：同梱 upstream の文書であり、ViewPulse の引き継ぎ文書とは分けて扱う

## 変更時の前提

- 端末内ファイル、カメラ映像、分析値を外部送信しない既存の境界を維持する。YouTube の公式プレイヤー接続は別の外部境界として扱う。
- 表情・視線は推定値であり、専用機器の測定値や感情の確定診断として扱わない。
- YouTube は URL・動画 ID・分析値だけを保存し、埋め込み映像のフレーム取得や動的 AOI を行わない。
- 動的 AOI は実験機能で、初期 OFF・測定後処理・名称確認を維持する。
- `view-pulse-ios` は別リポジトリのネイティブ版であり、Web 版の入力・カメラ構成と混同しない。

## 検証と記録

- `node --check app.js`、`node --check local-server.mjs`、`node verify.mjs` を基本検証とする。
- 変更時は、画像・動画・YouTube、カメラ同意、調整、記録、結果表示、ライブラリの主要経路を確認する。
- 変更後は `git diff --check` を実行する。
- 意味のある変更では `docs/current-state.md` と `docs/dev/logs/` のログを更新する。
