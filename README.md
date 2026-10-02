# Meeting Scribe

Zoom、Google Meet、Teams などのオンライン会議をローカルで文字起こしする macOS アプリです。macOS 15 以降の Apple Silicon を対象にしています。

## 仕様

- ScreenCaptureKit でマイクと Mac のシステム音声を別々に取得し、「自分」「相手」の2ラベルで表示します。
- 日本語モデル Kotoba-Whisper v2 Q5_0（約538 MB）を初回にダウンロードし、whisper.cpp の Metal 対応 `whisper-cli` で端末内推論します。会議音声と文字起こしを外部サービスへ送信しません。
- 結果は時刻、話者ラベル付きの UTF-8 テキストとして保存できます。生音声は記録終了後に一時ファイルから削除します。
- Zoom/Meet/Teams の会議らしいタイトルを持つウィンドウを確認し、開始・終了前に確認ダイアログを出します。完全な会議状態検出ではなく、手動操作が基本です。

## ビルドと起動

必要なものは Xcode Command Line Tools（Swift コンパイラと macOS SDK）、CMake、Git です。アプリは arm64 の `.app` バンドルとして生成します。署名 identity がある環境では `MEETING_SCRIBE_CODESIGN_IDENTITY` に identity 名を指定できます。省略時は ad-hoc 署名です。

```sh
brew install cmake
./scripts/build-whisper-cli.sh
./scripts/build-app.sh
open "dist/Meeting Scribe.app"
```

初回起動後、「実行ファイルを選択…」を押し、プロジェクト内の `.local/whisper.cpp/build/bin/whisper-cli` を選択します。「モデルを取得」でモデルを保存します。モデルは `~/Library/Application Support/MeetingScribe/Models/` に置かれます。準備後はネットワークを切っても文字起こしできます。

初回の録音開始時に macOS の「プライバシーとセキュリティ」で「画面収録とシステムオーディオ録音」と「マイク」のアクセスを Meeting Scribe に許可してください。アプリは開始時にmacOSのマイク許可を要求します。ScreenCaptureKit のシステム音声取得は画面収録権限を必要とします。開始時にTCCの拒否エラーが出る場合は、システム設定で両方のアクセスを許可してからアプリを再起動してください。

開発用ビルドは ad-hoc 署名です。Apple DTSの説明のとおり、再ビルド後は画面収録のTCC許可が新しいコード署名に引き継がれないことがあります。その場合は最終ビルドを起動し、システム設定で Meeting Scribe の画面収録許可を付け直してからアプリを再起動してください。安定した開発運用では Apple Development 証明書で署名してください。

## 話者ラベルと制約

「自分」はローカルのマイク入力、「相手」は Mac のシステム出力音声です。システム音声を再生する Zoom / Meet / Teams 以外のアプリ音声も混ざります。ブラウザーの場合、Meet のタブだけでなくブラウザーからの音声が対象になる場合があります。

ヘッドホンを使ってください。スピーカー音がマイクへ回り込むと、相手の発話が「自分」にも重複する場合があります。マイクの取得は会議アプリのミュート状態とは連動しません。録音中はローカルマイクに入った音声を取り込みます。

会議検知はウィンドウ名に基づくヒューリスティックです。会議開始・終了を正しく判定できない場合があるため、誤検知時は「後で」を選び、必要に応じて手動で開始・終了してください。無音状態だけでは自動終了しません。

## モデルとライセンス

Kotoba-Whisper v2 GGML Q5_0 モデルは [Apache-2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml) です。whisper.cpp は [MIT License](https://github.com/ggml-org/whisper.cpp) です。アプリ本体はこのワークスペース内のコードです。
