from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer


@dataclass
class Config:
    """設定."""

    host: str
    port: int
    debug: bool


def parse_args() -> Config:
    """引数を解析する.

    Returns:
        Config: 設定.
    """
    parser = argparse.ArgumentParser(description="vLLM互換モックサーバー")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return Config(host=args.host, port=args.port, debug=args.debug)


class MockHandler(BaseHTTPRequestHandler):
    """モックハンドラ."""

    server_version = "MockVLLM/0.1"

    def do_POST(self) -> None:  # noqa: N802
        """POSTを処理する."""
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
        prompt_text = ""
        messages = payload.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            for part in content:
                if part.get("type") == "text":
                    prompt_text = part.get("text", "")
                    break
        response_text = (
            "A calm looping scene with a single subject in a clear setting."
        )
        if prompt_text:
            response_text = f"{response_text} Prompt: {prompt_text[:60]}"
        response = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """ログを抑制する."""
        if getattr(self.server, "debug", False):
            super().log_message(format, *args)


def main() -> None:
    """エントリポイント."""
    config = parse_args()
    server = HTTPServer((config.host, config.port), MockHandler)
    server.debug = config.debug
    print(f"mock vllm server: http://{config.host}:{config.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
