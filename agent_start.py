"""Startup script for Gaokao-Master.

Examples:
    python agent_start.py web --port 8501
    python agent_start.py cli
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Gaokao-Master.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    web_parser = subparsers.add_parser("web", help="Start the Streamlit WebUI.")
    web_parser.add_argument("--port", type=int, default=8501)

    subparsers.add_parser("cli", help="Run a simple local CLI smoke test.")

    args = parser.parse_args()
    os.environ["PYTHONPATH"] = str(SRC_ROOT)

    if args.command == "web":
        start_web(args.port)
        return

    if args.command == "cli":
        start_cli()


def start_web(port: int) -> None:
    app_path = SRC_ROOT / "gaokao_master" / "web" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
    ]
    print(f"Starting Gaokao-Master WebUI: http://127.0.0.1:{port}")
    subprocess.run(command, check=True)


def start_cli() -> None:
    sys.path.insert(0, str(SRC_ROOT))

    from gaokao_master.agents import MainGaokaoAgent
    from gaokao_master.kb import KnowledgeBaseManager

    agent = MainGaokaoAgent(KnowledgeBaseManager())
    print("Gaokao-Master CLI 已启动。输入问题进行检索，输入 exit 退出。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        response = agent.invoke(user_input, top_k=5)
        print(f"\nAgent：{response.message}")
        for index, hit in enumerate(response.retrieval_hits, start=1):
            preview = hit.text.replace("\n", " ")[:180]
            print(f"{index}. [{hit.score:.3f}] {hit.subject}/{hit.topic} {preview}")


if __name__ == "__main__":
    main()
