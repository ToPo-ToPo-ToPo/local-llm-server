from __future__ import annotations

import argparse
import os
import sys
import threading

from .constants import DEFAULT_MODEL, DEFAULT_VISION_MODEL, _env_bool
from .router import RouterServer
from .server import (
    BACKENDS,
    DEFAULT_BACKEND,
    ServerConfig,
    ServerPool,
    build_command,
    build_pool_configs,
    find_pids_on_port,
    install_shutdown_handlers,
    parallel_supported,
    stop_pid,
)

# CLI で選べるバックエンド。router は内部でテキスト LLM と vision VLM を
# 同時起動し、リクエスト内容で自動振り分けする特別モード。
ROUTER = "router"
CLI_BACKENDS = (*BACKENDS, ROUTER)


def _run_router(args, extra: list[str], enable_thinking: bool) -> int:
    """router モード: テキスト LLM と vision VLM を同時起動し、プロキシで振り分ける。

    公開ポート(args.port)にルーターを立て、テキストを port+1、vision を port+2 で
    起動する。クライアントは公開ポートの 1 つだけを base_url に指定すればよい。
    vision バックエンドは mlx-vlm（Apple Silicon 前提）、テキストは mlx を使う。
    """
    text_port = args.port + 1
    vision_port = args.port + 2
    text_cfg = ServerConfig(
        backend="mlx",
        model=args.model,
        host=args.host,
        port=text_port,
        disable_thinking=not enable_thinking,
        extra_args=extra,
    )
    vision_cfg = ServerConfig(
        backend="mlx-vlm",
        model=args.vision_model,
        host=args.host,
        port=vision_port,
    )
    pool = ServerPool([text_cfg, vision_cfg])
    print(f"Starting (text):   {' '.join(build_command(text_cfg))}", file=sys.stderr)
    print(f"Starting (vision): {' '.join(build_command(vision_cfg))}", file=sys.stderr)
    try:
        pool.start()
        pool.wait_until_ready()
    except (RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        pool.stop()
        return 1

    router = RouterServer(
        (args.host, args.port),
        (args.host, text_port),
        (args.host, vision_port),
    )
    public_url = f"http://{args.host}:{args.port}/v1"
    print("Ready (automatic routing):", file=sys.stderr)
    print(f"  - public: {public_url}", file=sys.stderr)
    print(f"      text -> http://{args.host}:{text_port}/v1", file=sys.stderr)
    print(f"      image/media -> http://{args.host}:{vision_port}/v1", file=sys.stderr)
    print(
        f'In agent.toml, set only the single base_url = "{public_url}". '
        "Text-only requests are routed to the LLM, requests with images to the VLM, automatically.",
        file=sys.stderr,
    )
    print(
        "Note: if the VLM does not support function calling (tools), "
        'set tool_mode = "prompt" when using images.',
        file=sys.stderr,
    )

    thread = threading.Thread(target=router.serve_forever, daemon=True)
    thread.start()
    try:
        pool.wait()  # 子サーバーが終了するまでブロック
    except KeyboardInterrupt:
        pass
    finally:
        router.shutdown()
        router.server_close()
        pool.stop()
    return 0


def _stop_servers(ports: list[int]) -> int:
    """指定ポートで動いているローカルサーバーを探して停止する（--stop 用）。

    各ポートを LISTEN しているプロセスを lsof で特定し、プロセスグループごと
    SIGTERM→SIGKILL で止める。1つでも止めれば 0、見つからなければ 1 を返す。
    """
    if os.name != "posix":
        print(
            "--stop is supported on macOS / Linux only. On Windows, stop the server "
            "from its own window (Ctrl+C) or via Task Manager.",
            file=sys.stderr,
        )
        return 1
    stopped = False
    for port in ports:
        pids = find_pids_on_port(port)
        for pid in pids:
            print(f"Stopping the server on port {port} (pid {pid})...", file=sys.stderr)
            if stop_pid(pid):
                stopped = True
    if not stopped:
        ports_str = ", ".join(str(p) for p in ports)
        print(f"No running server found on port(s): {ports_str}.", file=sys.stderr)
        return 1
    print("Stopped.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-llm-server",
        description=(
            "Start a local LLM server (mlx / mlx-vlm / llama.cpp / router). "
            "Use the vision-capable mlx-vlm for image input (--backend mlx-vlm). "
            "With --backend router, start a text LLM and a VLM together and route requests automatically based on their content."
        ),
        epilog="Backend-specific extra arguments can be passed after --.",
    )
    parser.add_argument(
        "--backend",
        choices=CLI_BACKENDS,
        default=os.environ.get("CODER_BACKEND", DEFAULT_BACKEND),
        help="Backend to use (env: CODER_BACKEND, default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CODER_MODEL", DEFAULT_MODEL),
        help="Model (mlx: name/path, llama-cpp: .gguf path; used as the text model in router mode). env: CODER_MODEL, default: %(default)s",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("CODER_VISION_MODEL", DEFAULT_VISION_MODEL),
        help="Vision model that handles images/media in router mode. env: CODER_VISION_MODEL, default: %(default)s",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to (the first port when using a pool)")
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a server already running on --port instead of starting one "
        "(use the same --port / --instances / --backend router as when you started it). macOS / Linux only.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of concurrent processing slots (llama.cpp only; mlx processes sequentially)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help="Number of servers to start on consecutive ports (to gain parallelism with mlx; each loads its own model)",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Thinking mode. Use --no-thinking to disable (env: CODER_ENABLE_THINKING)",
    )
    parser.add_argument(
        "--draft-model",
        default=os.environ.get("CODER_DRAFT_MODEL"),
        help="MTP drafter for Gemma 4 (HF id/path, e.g. "
        "mlx-community/gemma-4-E4B-it-qat-assistant-bf16). Use 'auto' to select automatically from the main model name. "
        "Speeds up inference without changing the main model's output. mlx-vlm backend only. env: CODER_DRAFT_MODEL",
    )
    args, extra = parser.parse_known_args(argv)

    if args.instances < 1:
        parser.error("--instances must be 1 or greater")

    # --stop: 起動の代わりに、起動時と同じポート割り当てを再現して停止する。
    # router は public(port) / text(port+1) / vision(port+2)、通常は連番ポート。
    if args.stop:
        if args.backend == ROUTER:
            ports = [args.port, args.port + 1, args.port + 2]
        else:
            ports = [args.port + i for i in range(args.instances)]
        return _stop_servers(ports)

    # kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally（pool.stop）を
    # 必ず通し、起動したバックエンドを孫プロセスとして残さない。
    install_shutdown_handlers()

    if args.parallel is not None:
        if args.parallel < 1:
            parser.error("--parallel must be 1 or greater")
        if not parallel_supported(args.backend):
            print(
                f"Warning: {args.backend} does not support parallel slots "
                "(sequential processing). --parallel is ignored.",
                file=sys.stderr,
            )

    if args.backend == "mlx-vlm":
        print(
            "Info: starting with the vision backend (mlx-vlm). It supports image input (image_url), "
            "but some models do not support function calling (tools)/streaming. "
            'In that case, set tool_mode = "prompt" in agent.toml.',
            file=sys.stderr,
        )

    enable_thinking = (
        args.thinking if args.thinking is not None
        else _env_bool("CODER_ENABLE_THINKING", False)
    )

    # argparse が拾った先頭の "--" を除去してバックエンドへ素通し
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.backend == ROUTER:
        if args.instances != 1:
            parser.error("--instances cannot be used with --backend router")
        return _run_router(args, extra, enable_thinking)

    base = ServerConfig(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=args.port,
        parallel=args.parallel,
        disable_thinking=not enable_thinking,
        draft_model=args.draft_model,
        extra_args=extra,
    )
    configs = build_pool_configs(base, args.instances)
    pool = ServerPool(configs)

    for config in configs:
        print(f"Starting: {' '.join(build_command(config))}", file=sys.stderr)
    try:
        pool.start()
        pool.wait_until_ready()
    except (RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        pool.stop()
        return 1

    print("Ready:", file=sys.stderr)
    for url in pool.base_urls:
        print(f"  - {url}", file=sys.stderr)
    if args.instances > 1:
        print(
            "Point each OpenAI-compatible client at one of the base URLs above. "
            f'Example: base_url = "{pool.base_urls[0]}"',
            file=sys.stderr,
        )
    else:
        print(
            "Point your OpenAI-compatible client at "
            f'base_url = "{pool.base_urls[0]}".',
            file=sys.stderr,
        )

    try:
        pool.wait()
    except KeyboardInterrupt:
        pass
    finally:
        pool.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
