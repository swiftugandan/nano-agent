#!/usr/bin/env python3
"""
Live test for headless gateway: connect to agent --gateway ADDR, send
agent.run_turn, print JSON-RPC notifications and final result.
Usage: python scripts/test_gateway_ws.py [ws://127.0.0.1:8765]
"""
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets", file=sys.stderr)
    sys.exit(1)


async def run(url: str, prompt: str = "Reply with exactly: OK") -> None:
    async with websockets.connect(url, close_timeout=2) as ws:
        req = {
            "jsonrpc": "2.0",
            "id": "live-test-1",
            "method": "agent.run_turn",
            "params": {"prompt": prompt, "include_bus_events": False},
        }
        await ws.send(json.dumps(req))
        print("Sent:", json.dumps(req, indent=2))
        print("---")
        count = 0
        while count < 100:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120.0)
            except asyncio.TimeoutError:
                print("(timeout)")
                break
            count += 1
            try:
                obj = json.loads(msg)
                print(json.dumps(obj, indent=2))
            except Exception:
                print(msg)
            if "result" in obj and "assistant_text" in obj.get("result", {}):
                break
            if "error" in obj:
                break


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    asyncio.run(run(url))


if __name__ == "__main__":
    main()
