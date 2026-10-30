# 起動・運用・アンインストール

標準のuvプロジェクト環境で運用する。この文書の`gw`コマンドは、リポジトリで
`uv run --frozen gw ...`として実行する（または`.venv`をactivateして実行する）。

## インストール

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
uv sync --frozen
uv run --frozen gw enable     # 自動起動を使う場合
uv run --frozen gw status
```

`.venv`はuvが作る実体のあるディレクトリー。Python実行ファイルなどの内部はuv標準に任せる。
常駐サービスも、この`.venv/bin/gw`を直接登録する。グローバルな起動リンクは作らない。
`make install`は`uv sync --frozen`と`uv run --frozen gw enable`を実行する補助。
依存は`uv.lock`に従って同じ`.venv`へ同期し、通常の更新では世代別ディレクトリーを作らない。

### uv syncの使い分け

| 用途 | コマンド | ロックの扱い |
|---|---|---|
| 公開版の導入・利用環境の同期 | `uv sync --frozen` | 同梱の`uv.lock`を更新せず使う |
| 依存を追加・変更する開発作業 | `uv sync` | 必要に応じて`uv.lock`も更新する |

利用時はソースと`uv.lock`を同じ公開版に揃え、`--frozen`でその固定内容を使う。
`--frozen`はこの実行でロックを更新しない指定であり、環境を永久に固定する設定ではない。
また、`pyproject.toml`とロックの整合性確認は省略するため、依存定義を編集した後は
`uv sync`でロックを同期して差分を確認する。通常の`uv sync`も、既存ロックが最新ならその固定内容を使い、
毎回すべてのパッケージを最新版へ更新するわけではない。

`uv run`にも`--frozen`を指定できる。利用時の起動例も`uv run --frozen gw ...`に揃える。
どちらもuv標準の使い方で、実体の`.venv`を使う構成や証明書対策は変わらない。
詳細は[uv公式のロックと同期の説明](https://docs.astral.sh/uv/concepts/projects/sync/)を参照。

### 以前のシンボリックリンク構成からの移行

`.venv`がシンボリックリンクの場合は、リンクだけを外してから通常のuvで作り直す。
リンク先の旧環境はこの操作では削除されない。仮想環境は絶対パスを含むため移動しない。

```bash
if [ -L .venv ]; then unlink .venv; fi
uv sync --frozen
uv run --frozen gw enable
```

古いPythonでは新しい`gw`が起動を拒むため、最初の`uv sync`には修正済みuv 0.13.6以上を使う。
Pythonは`.python-version`の3.13.18を使う。共有PythonやPC全体のPATH・シェル設定は変更しない。
モデル・設定は従来の場所を使い、移行後の起動と機能を確認してから旧環境を整理する。
旧tool環境用の`scripts/install_runtime.py`と2世代保持処理は互換用に残すが、通常の導入では使わない。

依存を変更する開発作業では、通常のuvでロックも同期する。

```bash
uv sync
uv run pytest
make check
```

結線テスト用local-llm-client 0.10.3の依存もuv.lockで管理するため、手動の補完インストールは不要。
GitHub CIは使用していないため、PR前にmake checkでlint・型・セキュリティ・依存監査・テストを回す。

## 自動起動（既定。Ollama と同じ「サーバーを意識しない」運用）

`make install`（または `gw enable`）で、gw は **OS のユーザーサービス**として登録される
（macOS: launchd LaunchAgent / Linux: systemd user unit。Windows は未対応で従来どおり手動）。

- **ログイン時に自動起動**する。常駐してもモデル未使用時のコストはほぼゼロ
  （モデルは初回リクエストで遅延ロードされ、idle で自動アンロードされるため）。
- **異常終了（クラッシュ・kill -9）のときだけ自動復活**する。`gw stop` は正常終了なので
  復活しない——手動停止の意思を OS が上書きすることはない（再開は `gw start` か次回ログイン）。
  クラッシュループはスロットリング（launchd `ThrottleInterval` / systemd `StartLimit*`）で頭打ち。
- **ゴミは残らない**。ワーカー（モデルサーバー）はデーモンへ「繋留」されており、デーモンが
  どんな死に方をしてもワーカーは自分で終了する（パイプ EOF 検知）。さらにデーモンは起動時に
  ワーカー台帳を照合して前回の残骸を回収してから立ち上がる（crash-only: 起動処理＝復旧処理）。
- **管理者は OS の 1 者に固定**。`gw enable` は手動起動中のデーモンが居れば止めてから切り替える。
  単一起動（`GatewayLock`）は従来どおりで、真実は常に `gw status` の 1 箇所。
- 戻すのも 1 コマンド: **`gw disable`** で登録を解除し、完全に手動運用（`gw start`）へ戻る。
- サービスの実行口は `gw serve`（フォアグラウンド実行。Ollama の `ollama serve` 相当）。
  端末で直接回してデバッグしたいときにも使える。サービスの標準出力は
  `~/.cache/local-llm-server/logs/service.log` へ（通常のログは従来どおり `gw log`）。
- **メニューバーアイコン（macOS）**: 稼働中はメニューバーに「gw」が出る（Ollama 流の
  「動いていることが見える」）。アイコンはデーモンに繋留されているので、**アイコンの存在＝
  デーモンの生存**（kill -9 でもアイコンだけ残らない）。状態（接続先 URL・ロード中モデル）
  は**メニューを開いた瞬間にだけ**取得する——Ollama と同じく常駐中の定期処理は持たず、
  アイドル時の CPU 使用はゼロ。メニューからログを開く・停止もできる。
  `gateway.toml` の `tray = false` で無効化（→ docs/gateway.md）。
- **更新アイコンは通知専用**: 新版を検知するとアイコンが「gw ⬆」に変わり、メニューに
  利用可能な版と「ターミナルで `gw update`」を表示する。メニューから更新は実行しない。
  **放置しても更新は適用されない**。自動で行うのは確認とアイコン通知だけで、適用は
  ターミナルで `gw update` を明示的に実行したときだけ。
  更新が済めば**マークは降りる**——別のターミナルで `gw update` / `git pull` した場合も、
  次にメニューを開いたときの確認で消える。`git pull` 済みでデーモンだけが古いコードの
  ままなら（要再起動）マークは残り、`gw update` で再起動して解消する。
  メニューを開いた際の更新確認は裏で行い、結果は開いている更新項目へその場で反映する
  （メニューを閉じてもう一度押す必要はない）。

## 起動と状態確認（リポジトリ内）

```bash
uv run --frozen gw status     # 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
uv run --frozen gw ps         # ロード中モデルの状態（処理中数・在席・アイドル残り）
uv run --frozen gw start      # 手動起動（自動起動を無効にしている場合や、gw stop からの再開に）
```

設定ファイルは **`~/.config/local-llm-server/gateway.toml` の 1 箇所だけ**。無ければ初回の
`gw start` が自動生成する（クローンの例 gateway.toml を複製。以降の編集・参照はこの 1 ファイルのみ）。
引数なしの `gw` はコマンド一覧を表示する（起動の入口は `gw start` の 1 つだけ）。

## サブコマンド一覧

| コマンド | 動作 |
|---|---|
| `gw start` | デーモンを裏で常駐起動（既存デーモンは管理 API の応答を確認して再利用。起動途中は待機し、応答しなければ明示エラー） |
| `gw stop` | このパッケージ由来のゲートウェイ／モデルサーバーを停止 |
| `gw restart` | stop → start |
| `gw enable` | ログイン時自動起動＋異常終了時自動復活を OS に登録（`make install` が実行済み） |
| `gw disable` | 自動起動を解除して手動運用（`gw start`）に戻す |
| `gw serve [--managed]` | フォアグラウンドで実行（サービスの実行口・デバッグ用） |
| `gw status` | 稼働/停止を 1 行で表示（停止中は終了コード 1） |
| `gw ps` | ロード中モデルの状態を表示 |
| `gw list` | 使えるモデル一覧（カタログ＋HF キャッシュ。停止中でも表示可） |
| `gw pull <model>` | モデルを事前ダウンロード（進捗付き。GGUF は `org/repo:量子化名` で本体+mmproj のみ） |
| `gw rm <model>` | モデルを HF キャッシュから削除（確認あり・ロード中は拒否。**自動では決して走らない**） |
| `gw show <model>` | モデルの素性を表示（量子化・コンテキスト長・サイズ・画像入力・MTP・ロード状態） |
| `gw log [-f] [-n N]` | ゲートウェイログの末尾を表示（`-f` で追従） |
| `gw max <n\|off>` | `max_resident` を無停止で変更（超過はアイドルから LRU 退避） |
| `gw mtp [model]` | MTP に必要なドラフターと取得状況を表示（ダウンロードはしない → docs/mtp.md） |
| `gw update` | 新しいリリースタグがあれば追従し、稼働中なら再起動（設定の移行も込み）。新版が無くても、稼働中デーモンが古いコードのままなら再起動する |
| `gw migrate [--dry-run]` | 設定ファイルを現行スキーマへ移行（**通常は更新時に自動**。`--dry-run` で確認だけ） |
| `gw help` | サブコマンド一覧を表示（設定ファイル不要） |

- **どこからでも動く**: どのディレクトリから打っても、設定は `~/.config` の 1 ファイル、デーモンは
  マシンに 1 つ（`GatewayLock`）なので、常に同じものに届く。`status`/`stop`/`ps`/`list`/`log`/
  `max`/`update` は**稼働中デーモンのランタイム記録**（temp の `local-llm-server-gateway.json`：
  host/port/PID）を最優先で辿るため、実際に動いているデーモンを見失わない。記録は正常停止で消え、
  クラッシュで残っても PID 生存チェックで stale を掴まない。
- ログは `~/.cache/local-llm-server/logs/`（Ollama の `~/.ollama/logs` 相当の固定パス。
  **どこから起動しても同じ場所**で、起動したディレクトリには何も作らない）。
  `gw log` で末尾を表示できる。
- `gw list` の一覧には**ロード中だけでなく、HF キャッシュにある DL 済みモデルも未ロード候補**として並ぶ。
- **更新の確認と通知**は稼働中デーモンが裏で行う。新しいリリースタグを検知すると、
  macOS のメニューバーアイコンに「⬆」を表示する。この常駐処理は `git fetch` / `git merge` / `uv sync`
  や再起動を実行しない。適用はターミナルの `gw update` だけ。
  旧 `auto_update` 設定は廃止済み。残っていても起動は止まらず、更新後の初回起動で自動移行が
  その行を理由付きのコメントへ書き換える（手で消す必要はない）。
- **「pull 済みだがプロセスが古い」を検知する**: editable 運用（`make install`）では
  `git pull` した瞬間にディスク上のソースだけが新しくなり、稼働中デーモンは pull 前のコードを
  保持し続ける。更新判定はソース版を見るので「もう最新」と結論し、放っておくと**新版が永久に
  効かない**。これを防ぐため、デーモンは起動時に「自分が読み込んだコードの版」を記録し、
  ディスク上のソース版と食い違ったら `restart_required` を立てる（`/admin/status` の `update`
  に `restart_required` と `running` が出る）。`gw update` はこのとき、取ってくるものが無くても
  再起動する。オフラインでも判定できる（リリースタグとの比較とは独立）。
- **設定は更新に自動追従する**: 新版でキーが廃止・改名されると、`~/.config` の `gateway.toml` は
  そのままでは「書いてあるのに効かない」旧キーを抱えることになる。これを避けるため、更新後の
  起動時（`gw start`）と `gw update` で**設定ファイルを実際に書き換える**。
  コメントと並びは保ち、書き換える前に `gateway.toml.bak` を残す。手で確認・実行したいときは
  `gw migrate`（`--dry-run` で差分だけ表示）。廃止キーの一覧は `local_llm_server/migrate.py` の
  `OBSOLETE_KEYS`。
- **リリース手順（開発者向け）**: `pyproject.toml` の version を上げて main へマージし、タグを push する。

  ```bash
  git tag v<新バージョン>
  git push origin v<新バージョン>
  ```

  各 PC の常駐デーモンが 1 時間以内にタグを検知し、macOS ではアイコンに「⬆」を表示する。
  適用する PC のターミナルで `gw update` を手動実行する。

- **更新の「確認」はリスタート不要**（Ollama と同じく、確認は動いたまま・**適用のときだけ**再起動）。
  定期チェックは 1 時間おきだが、それを待たずとも**メニューバーのメニューを開くたびに
  その場で確認**が走る（`/admin/status` がオンデマンドでチェックを温める。30 秒スロットル）。
  `git pull` した直後でも、メニューを開けば「更新の有無」が最新化される。

## アンインストール

`make uninstall`で自動起動の登録とアプリ設定・ログを削除する。プロジェクトの.venvとモデル重みは保持する。

```bash
cd local-llm-server
make uninstall   # プロジェクトのgwで停止・登録解除し、設定・ログ・llama.cppを削除
```

消えるのはクローン外に残る次のもの（クローン自体は手動で削除する）。

| 場所 | 中身 |
|---|---|
| `~/.config/local-llm-server/` | `gateway.toml` |
| `~/.cache/local-llm-server/` | ログ（`logs/`）と自動ダウンロードした `llama.cpp` バイナリ |
| `~/Library/LaunchAgents/com.local-llm-server.gw.plist` ほか | 自動起動の登録（`gw disable` が削除。Linux は `~/.config/systemd/user/local-llm-server.service`） |

ログもキャッシュも **cwd 非依存の固定パス**に書くので、どのディレクトリから `gw start` しても
この 2 つの外には何も残らない。

### ログは自動で頭打ちになる

`gw start` のたびにログを整理するので、放置しても肥大化しない（Ollama と同じ 5 世代）。

- **デーモン本体のログ**: `gateway-<port>.log` → `gateway-<port>-1.log` … と押し出し、
  5 世代を超えた最古を削除する。
- **モデルサーバーのログ**: ファイル名にポート番号が入る（`server-9001.log`）ため世代を
  押し出せない——ポートが変わるたびに別ファイルが増える。そこで**新しい順に 5 個だけ残して**
  古いものを削除する。刈るのはゲートウェイ起動時だけなので、書き込み中のログは消えない。

サイズ上限は設けていない。書かれるのは起動・停止・モデルのロード/破棄といったイベント行だけで、
リクエストごとには書かないため、世代数の上限だけで十分に頭打ちになる。

`gw stop` は同パッケージ由来のプロセスだけを止め、同じポートをたまたま使う無関係なプロセスは
巻き添えにしない。親 PID だけでなく同じプロセスグループの子孫まで終了確認し、死亡を確認できた
ワーカーだけを台帳から削除する。停止を確認できなかった記録は次回起動時の回収用に保持する。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の共有キャッシュ
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため**アンインストールでは一切
> 削除しない**（場所を案内するだけ）。不要になったら自分で中のモデルフォルダを消す。


## TLSと実行環境

開発環境は`.python-version`のCPython 3.13.18に固定する。`make dev`、`make check`、
`uv run --frozen …`を使用する。uvは0.13.6以上を必要とする。
リポジトリ専用uvを使う場合の配置先は`.tools/uv/0.13.6/uv`（Windowsはuv.exe）。
公式配布物を取得し、公式リリースのハッシュ・署名情報を確認して配置する。
既存の信頼ストアが未修正のツールでブートストラップしないこと。
共有uv、シェル設定、PC既定PythonやOS証明書ストアの変更は不要。

公開パッケージの最小Pythonは系列ごとに3.11.18、3.12.16、3.13.18、3.14.10、
3.15.2。既知の未修正パッチをRequires-Pythonでも除外する。certifi>=2026.10.19、
pipを使う環境ではpip>=26.2.2を必要とする。vLLM 0.28.0の追加機能は、固定された
GPU依存wheelが対応するLinux / Python<3.15に限定する。vLLM自体の固定版は維持する。
参照: [Python公式](https://blog.python.org/2026/10/python-3152-31410-31318-31216-31118/)、
[pip](https://pip.pypa.io/en/stable/news/)、[uv](https://github.com/astral-sh/uv/releases/tag/0.13.6)。

`make install`は標準の`uv sync --frozen`でプロジェクトの`.venv`を同期し、
`uv run --frozen gw enable`で自動起動を登録する。依存の固定版はuv.lock、実行環境は.venvに集約する。
旧tool導入専用の世代切り替え・自動整理は通常のプロジェクト環境には適用しない。
追加の任意extrasを使う場合は、通常のuvと同様に`uv sync --frozen --extra <名前>`で指定する。
その機能を使って起動する例は`uv run --frozen --extra <名前> gw start`。
Windowsでも標準のuvで環境を作り、`uv run --frozen gw start`で手動起動する（自動起動サービスは未対応）。

`gw pull`と`./scripts/hf`は、hf-xetの独立したRust TLS経路を迂回する。
このアプリ内でHF_HUB_DISABLE_XET=1を設定するが、TLS証明書検証は有効のまま。
別途直接起動した`hf`や他アプリには適用されない。実験Strata Liteも同じ迂回を行う。
Gitの更新通信とテンプレート取得curlには、このアプリの修正済みCAバンドルを指定する。
企業独自CAを利用する環境ではCAの運用要件も個別に確認する。

TLS検査用のチェーン資料を[説明サイト](https://emudhra.we-have-a-problem.com/)から取得した後、
以下で正常対照と14ホストを検査できる。取得したスクリプトを実行する必要はない。

```bash
uv run --frozen python scripts/check_emudhra.py --reference-chains /path/to/chains.zip
```

各実装でjulialang.orgの成功を先に確認する。認証情報なし、リダイレクトなし、TLS検証有効で実施し、
エラーの具体的理由と証明書指紋を記録する。DNS/タイムアウト、期限切れ、ホスト名不一致、CT要件だけの
拒否を、侵害CA拒否とは混同しない。CI/Docker定義は本リポジトリに存在しないため、
将来導入する際はPython・uv・CA・独自TLSをそれぞれ更新し、コンテナ内でも同じ検査を行う。

VS Codeはリポジトリ内`.vscode`のPythonパスを使用する。以前に別Pythonを選択済みの場合は
「Python: Select Interpreter」でも`.venv/bin/python`を選択する。PC共通設定は変更しない。
Python本体を変更しても、IDE/Electronやllama.cpp独自TLSの更新確認の代わりにはならない。

通常の復元ではソースとuv.lockを対象リリースに合わせ、停止後に`uv sync --frozen`してから
`uv run --frozen gw enable`で登録し直す。仮想環境フォルダーの移動・改名は行わない。
セキュリティ対策前のPythonや依存へ戻す場合は、その環境での外部通信を安全確認済みとは扱わない。


結線テスト用local-llm-client 0.10.3のhttpx等はuv.lockに含まれ、通常のuv syncで導入される。
この固定ファイルは開発環境専用で、稼働用MLX等を更新しない。


Python 3.15はゲートウェイ本体・llama.cppの対象とする。Mac用MLX/torchvisionの
対応wheelが揃っていないため、MLX系の自動依存と既定選択はPython<3.15に限定する。
このリポジトリで固定するPython 3.13.18では、従来どおりMLXを使用できる。


### 追加の依存セキュリティ更新（2026-10-28）

公開要件はfilelock>=4.2.1、Pillow>=13.0.0、urllib3>=2.8.0、
macOSのsetuptools>=84.1.0。今回のMac用固定版は順に4.2.4、13.0.0、
2.8.0、84.2.1とする。ビルド用setuptoolsも>=84.1.0を要求する。
隔離バックエンドにも下限を適用し、導入済みでも古い版があれば別環境を使う。
setuptoolsの実行時下限はmacOSだけに適用する。LinuxのvLLM固定依存は
setuptools<81を要求するため、Linux/WindowsやCUDA実行を含めた全環境の
監査合格を意味しない。

Pillow 13では古い画像APIとIモードのPNG保存が変わり、filelock 4では
SoftReadWriteLockの保存形式が変わる。関連プロセスを止めて切り替え、
画像処理・ダウンロードの排他制御を検証する。今回のHF Hub経路は
FileLock/SoftFileLockを使う。無関係なMLX・transformers・torchの版は維持する。
検証済みの新環境へ参照を切り替え、本番用の導入環境は現行と直前の2世代を保持する。
Kokoroとmisakiは導入しない。

音声隔離環境のmlx-audio 0.5.5は版を維持するが、server extraは使わない。
このextraが要求するwebrtcvad 2.0.10はpkg_resourcesに依存し、修正済み
setuptoolsと両立しない。サーバーの他の依存は明示し、webrtcvadを
同じAPIを提供するwebrtcvad-wheels 2.0.14.post1へ置換する。
両方を同時導入すると同名モジュールが競合するため、同時導入しない。
Kokoro・misakiを追加する変更ではない。
