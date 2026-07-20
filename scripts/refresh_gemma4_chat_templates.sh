#!/usr/bin/env bash
#
# refresh_gemma4_chat_templates.sh
#
# 公式 google/gemma-4-*-it が 2026-07-15 に chat_template.jinja を更新した
# （null handling / reasoning(thinking) preservation / turn-tag・continuation の
#  書き換え / image_url・input_audio エイリアス / .get() による null 安全化）。
# このとき safetensors 重み・config・generation_config・tokenizer 類は
# すべて OID 完全一致で **無変更** だった。
#
# したがって MLX 版は再量子化不要で、各リポジトリの chat_template.jinja だけを
# 新公式に差し替えれば、フル再変換とビット単位で同一の結果になる。
# 本スクリプトは公式ソースから最新テンプレを取得し、ToPo-ToPo の各 MLX リポジトリへ
# 単一ファイル commit で push する（README はユーザ独自版のため上書きしない）。
#
# 前提: `hf auth whoami` が書き込み権のあるアカウント（ToPo-ToPo）でログイン済み。
# 使い方: bash scripts/refresh_gemma4_chat_templates.sh [--dry-run]
set -euo pipefail

DRY="${1:-}"
WORK="$(mktemp -d)"
MSG="chore: sync chat_template.jinja with upstream google/gemma-4 (2026-07-15: null handling, thinking preservation, turn-tag/continuation rewrite)"

# 公式ソース → 期待 OID（取得後に検証する）
LARGE_SRC="google/gemma-4-31B-it"                       # 4741bf6e...（31B/26B-A4B/12B 系で共通）
EFFICIENT_SRC="google/gemma-4-E2B-it-qat-q4_0-unquantized"  # fbe3b59b...（E2B/E4B 系で共通）
LARGE_OID="4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd"
EFFICIENT_OID="fbe3b59b625cd1b8850ea592d4203df6ec04684b"

curl -fsSL "https://huggingface.co/${LARGE_SRC}/resolve/main/chat_template.jinja"     -o "$WORK/large.jinja"
curl -fsSL "https://huggingface.co/${EFFICIENT_SRC}/resolve/main/chat_template.jinja" -o "$WORK/efficient.jinja"

got_large=$(git hash-object "$WORK/large.jinja")
got_eff=$(git hash-object "$WORK/efficient.jinja")
[ "$got_large" = "$LARGE_OID" ]     || { echo "ERROR: large template OID mismatch: $got_large" >&2; exit 1; }
[ "$got_eff"   = "$EFFICIENT_OID" ] || { echo "ERROR: efficient template OID mismatch: $got_eff" >&2; exit 1; }
echo "テンプレート取得・OID検証OK"

# MLX リポジトリ → 使用テンプレ（large|efficient）
REPOS=(
  "ToPo-ToPo/gemma-4-31b-it-mlx-4bit|large"
  "ToPo-ToPo/gemma-4-31b-it-mlx-8bit|large"
  "ToPo-ToPo/gemma-4-31b-it-mlx-bf16|large"
  "ToPo-ToPo/gemma-4-31b-it-qat-mlx-4bit|large"
  "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-4bit|large"
  "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-8bit|large"
  "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-bf16|large"
  "ToPo-ToPo/gemma-4-26B-A4B-it-qat-mlx-4bit|large"
  "ToPo-ToPo/gemma-4-E2B-it-qat-mlx-4bit|efficient"
  "ToPo-ToPo/gemma-4-E4B-it-qat-mlx-4bit|efficient"
)

for entry in "${REPOS[@]}"; do
  repo="${entry%%|*}"; kind="${entry##*|}"
  src="$WORK/${kind}.jinja"
  echo "=== $repo  (<- $kind template) ==="
  if [ "$DRY" = "--dry-run" ]; then
    echo "  [dry-run] hf upload $repo $src chat_template.jinja"
    continue
  fi
  hf upload "$repo" "$src" chat_template.jinja --commit-message "$MSG"
done

echo "完了。"
