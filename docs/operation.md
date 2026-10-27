# 起動・運用・アンインストール

運用は `gw` の **CLI サブコマンド**で行う。デーモン本体は端末を持たず裏で常駐し、`gw` は
「そのデーモンを起動・停止・監視する薄い CLI」。マシンに 1 ゲートウェイだけ（`GatewayLock`）なので、
`gw` を何度打ってもデーモンは 0 個か 1 個。

## インストール（一度だけ）

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
make install                     # これだけ。導入・PATH 設定・自動起動の登録まで全部やる
```

`make install` は `uv tool install --editable . --reinstall` に続けて次を全部行う
（editable は確定なので畳んである。再実行で入れ直しにもなる）:

1. **`/usr/local/bin/gw` の設置（要・管理者パスワード 1 回）** — Ollama と同じく、最初から
   PATH に入っている場所へ symlink を置く。これで **今のシェルからそのまま `gw` が使える**
   （exec や新ターミナル不要）。パスワードを入れずスキップしても壊れない（下記 2 が受け皿）
2. **PATH 設定（`uv tool update-shell`）** — `~/.local/bin` をシェル設定に追記
   （既に通っていれば何もしない）。1 をスキップした場合は新しいターミナルから有効
3. **`gw enable`（自動起動の登録）**

**editable** なのでソースはこのクローンを指し、`gw update` の取り込みがそのまま効く。

### 導入直後に `gw: command not found` になるとき

`/usr/local/bin` への設置をスキップした場合に、**今のシェルに PATH がまだ反映されていない**だけ
（環境変数は起動済みのシェルへ外から注入できない——これはどのツールでも同じ）。
新しいターミナルを開くか **`exec $SHELL -l`** で反映する（追記先は zsh では `~/.zshenv` なので
`source ~/.zshrc` では反映されない）。いつでも `make install` を再実行すれば
`/usr/local/bin` への設置からやり直せる。

> `~/.zshrc` に `. "$HOME/.local/bin/env"` がある環境では、そちら経由でも PATH が通る。これは
> uv を**スタンドアロンインストーラ**で入れたときに作られる行で、Homebrew 版の uv では作られない
> （環境によって有無が違うので、上の手順に頼るのが確実）。

開発（テストを回す）は `make dev`（＝`uv sync`）で開発用 venv を作り `uv run pytest`。
PR の前には `make check`（lint・型・セキュリティ・依存監査・テストと網羅率）を手元で回す。
GitHub の CI は使わない（ワークフローは置いていない）。
`make install` と併用してよい（ソースは共有され、`gw` は編集が即反映される）。

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

## 起動と状態確認（どのディレクトリからでも）

```bash
gw status     # 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
gw ps         # ロード中モデルの状態（処理中数・在席・アイドル残り）
gw start      # 手動起動（自動起動を無効にしている場合や、gw stop からの再開に）
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

`make install` の逆を 1 コマンドで実行すれば、自動起動の登録も含めて跡形なく消える。

```bash
cd local-llm-server
make uninstall   # gw disable → gw stop → uv tool uninstall → 設定・ログ・llama.cpp を削除
```

消えるのはクローン外に残る次のもの（クローン自体は手動で削除する）。

| 場所 | 中身 |
|---|---|
| `~/.config/local-llm-server/` | `gateway.toml` |
| `~/.cache/local-llm-server/` | ログ（`logs/`）と自動ダウンロードした `llama.cpp` バイナリ |
| `~/Library/LaunchAgents/com.local-llm-server.gw.plist` ほか | 自動起動の登録（`gw disable` が削除。Linux は `~/.config/systemd/user/local-llm-server.service`） |
| `/usr/local/bin/gw` | install が置いた symlink（リンク先が自分のときだけ削除。要・管理者パスワード） |

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
`./scripts/uv run --frozen …`を使用する。uvは0.13.6以上を必要とする。
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

`make install`は`scripts/install_runtime.py --activate`を実行し、`.runtime/installs/`
以下に新しいtool環境を構築する。旧環境は削除しない。既存環境がある場合はその導入版を
constraintsに保存し、MLX等を開発ロックに合わせて後退させない。新規導入では本体ロックを使う。
`install.json`、`constraints.txt`、環境内の`.gw-runtime-constraints.txt`が実際の固定版を示す。
更新時も稼働環境の固定版をconstraintsとして同期する。新要件と衝突した場合は失敗として
報告し、無関係な依存を一括最新版化しない。アプリの再起動後には実行プロセスのPythonも確認する。
Windowsは修正済みuvで`uv tool install --python 3.13.18 --editable . --with pip==26.2.2`
を使用する（上記サービス切替スクリプトはmacOS/Linux用）。既存環境の退避を先に行う。

`gw pull`と`./scripts/hf`は、hf-xetの独立したRust TLS経路を迂回する。
このアプリ内でHF_HUB_DISABLE_XET=1を設定するが、TLS証明書検証は有効のまま。
別途直接起動した`hf`や他アプリには適用されない。実験Strata Liteも同じ迂回を行う。
Gitの更新通信とテンプレート取得curlには、このアプリの修正済みCAバンドルを指定する。
企業独自CAを利用する環境ではCAの運用要件も個別に確認する。

TLS検査用のチェーン資料を[説明サイト](https://emudhra.we-have-a-problem.com/)から取得した後、
以下で正常対照と14ホストを検査できる。取得したスクリプトを実行する必要はない。

```bash
./scripts/uv run --frozen python scripts/check_emudhra.py --reference-chains /path/to/chains.zip
```

各実装でjulialang.orgの成功を先に確認する。認証情報なし、リダイレクトなし、TLS検証有効で実施し、
エラーの具体的理由と証明書指紋を記録する。DNS/タイムアウト、期限切れ、ホスト名不一致、CT要件だけの
拒否を、侵害CA拒否とは混同しない。CI/Docker定義は本リポジトリに存在しないため、
将来導入する際はPython・uv・CA・独自TLSをそれぞれ更新し、コンテナ内でも同じ検査を行う。

VS Codeはリポジトリ内`.vscode`のPythonパスを使用する。以前に別Pythonを選択済みの場合は
「Python: Select Interpreter」でも`.venv/bin/python`を選択する。PC共通設定は変更しない。
Python本体を変更しても、IDE/Electronやllama.cpp独自TLSの更新確認の代わりにはならない。

復元時は新プロセスを停止し、各install.jsonに記録した旧ランチャーと保存済みサービス定義を戻す。
開発用.venvを退避した場合は、そのディレクトリを元の.venvの位置に戻してから使う
（console scriptには作成時の絶対パスが入る）。旧環境には未修正TLSが残るため、復元は
短時間の障害切り分け用途とし、その環境での外部通信を安全確認済みとは扱わない。


結線テスト用の公開local-llm-clientはhttpxを未宣言でimportする版があるため、
`make dev` / `make check`は`requirements/test-client.lock`のハッシュ付き補完依存も導入する。
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
旧環境を削除せず、検証済みの新環境へ参照を切り替える。
Kokoroとmisakiは導入しない。

音声隔離環境のmlx-audio 0.5.5は版を維持するが、server extraは使わない。
このextraが要求するwebrtcvad 2.0.10はpkg_resourcesに依存し、修正済み
setuptoolsと両立しない。サーバーの他の依存は明示し、webrtcvadを
同じAPIを提供するwebrtcvad-wheels 2.0.14.post1へ置換する。
両方を同時導入すると同名モジュールが競合するため、同時導入しない。
Kokoro・misakiを追加する変更ではない。
