"""
Thin TCP client that sends bpy commands to the Blender socket addon.
All agents import and call BlenderBridge — they never talk to Blender directly.
"""

import socket
import json
import time


HOST = "127.0.0.1"
PORT = 9876
TIMEOUT = 30


class BlenderBridgeError(Exception):
    pass


class BlenderBridge:
    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self):
        if self._sock:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def reconnect(self, retries: int = 5, delay: float = 2.0):
        """Close current socket and attempt to reconnect (used after Blender restarts socket server)."""
        self.disconnect()
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                print(f"  [Bridge] Reconnecting... attempt {attempt}/{retries}")
                self.connect()
                print("  [Bridge] Reconnected.")
                return
            except OSError as e:
                last_err = e
                time.sleep(delay)
        raise BlenderBridgeError(f"Could not reconnect to Blender after {retries} attempts: {last_err}")

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def run(self, code: str) -> str:
        """
        Send a bpy code block to Blender and return the result string.
        Auto-reconnects once on broken pipe.
        Raises BlenderBridgeError on execution failure.
        """
        for attempt in range(2):
            try:
                self.connect()
                msg = json.dumps({"code": code}) + "\n"
                self._sock.sendall(msg.encode("utf-8"))

                raw = b""
                while b"\n" not in raw:
                    chunk = self._sock.recv(65536)
                    if not chunk:
                        raise BlenderBridgeError("Connection closed before response received")
                    raw += chunk

                line = raw.split(b"\n")[0]
                response = json.loads(line.decode("utf-8"))

                if response["status"] != "ok":
                    raise BlenderBridgeError(response.get("error", "Unknown error from Blender"))

                return response.get("result", "")

            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt == 0:
                    print("  [Bridge] Connection lost — reconnecting...")
                    self.reconnect()
                else:
                    raise BlenderBridgeError("Blender socket server is not available. Restart the addon in Blender.")

    def run_many(self, commands: list[str]) -> list[str]:
        results = []
        for cmd in commands:
            results.append(self.run(cmd))
        return results

    def clear_scene(self):
        self.run(
            "bpy.ops.object.select_all(action='SELECT')\n"
            "bpy.ops.object.delete(use_global=False)\n"
            "result = 'done'"
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
