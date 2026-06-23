# examples

ゲートウェイに接続して実LLMで生成する動作サンプル（**接続専用**。例自身はサーバーを起動しない）。
[PEP 723](https://peps.python.org/pep-0723/) のインライン依存を埋め込んであるので `uv run` で動く。

## 事前準備：ゲートウェイを起動する

別ターミナルで、`gateway.toml` に `mlx-community/Qwen3.6-27B-4bit`（`backend = "mlx-vlm"`、
速度を見るなら `draft_model = "auto"`）を登録して起動しておく:

```bash
uv run local-llm-server
```

リポジトリ直下の同梱 [`gateway.toml`](../gateway.toml) がそのまま使える。起動していないと例は
案内を出して終了する。

## サンプル

| ファイル | 内容 |
|---|---|
| `connect_and_generate.py` | ゲートウェイに接続してストリーミング生成（最小） |
| `generate_with_mtp.py` | 同上＋速度（tok/s）を表示。MTP は gateway.toml 側で設定 |

```bash
uv run examples/connect_and_generate.py
uv run examples/generate_with_mtp.py
```

MTP（投機的デコード）の詳細は [docs/mtp.md](../docs/mtp.md)、接続方法の詳細は
[docs/connecting.md](../docs/connecting.md) を参照。
