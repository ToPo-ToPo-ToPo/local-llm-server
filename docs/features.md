# 特徴と全体像

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）と音声認識（**whisper** / STT）を束ねる
**マルチモデルゲートウェイ**。1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **モデルの事前登録は不要**。クライアントが指定した `model` をその場でロードする。画像入力
  （mmproj 自動検出）も mlx-vlm の MTP も設定なしで効く。
- **画像・動画入力**。画像はそのまま、**動画（`video_url`）はゲートウェイが ffmpeg で等間隔に
  フレーム抽出して**モデルへ渡す（llama-cpp / mlx-vlm 共通。ffmpeg は pip 同梱で追加インストール
  不要 → [動画入力](gateway.md)）。
- **llama.cpp は自動導入**。Linux / Windows / Intel Mac では `llama-server` を OS・GPU 検出のうえ
  **起動時に自動ダウンロード**する（手動導入不要・導入方法の選択肢は無い一本道。GPU は Vulkan
  → [llama-cpp.md](llama-cpp.md)）。
- **vLLM / SGLang も選べる**（Linux/NVIDIA・Windows は WSL2）。`backend = "vllm"` または
  `"sglang"` で高スループット生成。現在の環境に有ればそれを使い、無ければ隔離 venv へ
  **起動時に自動導入**（SGLang は RadixAttention でエージェント用途に強い → [vllm.md](vllm.md)）。
- **音声認識（STT）も同じポートで**。`/v1/audio/transcriptions` に音声を投げれば mlx-whisper が
  遅延起動して文字起こしする。エージェント側に mlx 依存は要らない
  （→ [音声認識（STT / whisper）](gateway.md#音声認識stt--whisper)）。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **デーモンは裏で常駐、運用は `gw` の CLI サブコマンド**（Ollama 流）。`gw start` で常駐起動、
  `gw status`/`gw ps` で稼働確認、`gw stop` で停止。端末を占有しない。`status`/`stop` 等は
  **`gateway.toml` の無い場所からでも**唯一のデーモンを見つけて叩ける（→ [起動・運用](operation.md)）。
- **`gw list` が使えるモデルを自動一覧**。カタログに加え HF キャッシュの DL 済みモデルも
  未ロード候補として並ぶので、どれを指定すればよいか一目で分かる。
- モデルは**初回リクエスト時に遅延起動**、`max_resident`（数）/ `max_memory_fraction`（メモリ量）
  超過で LRU 退避、`idle_timeout` で自動アンロード。
- エージェントが「使い終わった」と通知すれば、在席が 0 になった瞬間に**待たず即アンロード**して
  メモリ解放（→ [在席ベースの即時アンロード](gateway.md#在席ベースの即時アンロード)）。
- **別PCからも接続できる**。`host = "0.0.0.0"` で LAN に公開し、`api_key` で認証
  （→ [別PCから接続する](gateway.md#別pcから接続するネットワーク公開)）。
- **自動更新**。clone 運用でも、常駐デーモンが PyPI 新版を検知して `git pull` で追従し新コードで
  再起動（作業ツリーがクリーンな時だけ・処理中/在席が空くのを待つ）。手動で今すぐなら `gw update`
  （→ [自動更新](gateway.md#自動更新pypi-新版に-git-で追従)）。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ
  [local-llm-client](https://pypi.org/project/local-llm-client/)）。

## 他 OS（Linux / Windows / Intel Mac）での動作

mlx は入らず、`llama-server` がゲートウェイ起動時に**自動でダウンロード・導入される**
（OS・CPU アーキ・GPU を検出し、GPU なら Vulkan・無ければ CPU を選択。PATH は汚さない）。
手動導入や PATH 設定は不要で、`gw start` して GGUF モデルの ID を投げるだけで動く。
検出の上書き（`accel`）とビルド固定（`pin`）は `gateway.toml` の `[llama_cpp]` で
（→ [llama-cpp.md](llama-cpp.md)）。
