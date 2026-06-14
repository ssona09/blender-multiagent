"""
Blender Socket Bridge Addon — thread-safe version.

The socket server runs in a background thread (safe for I/O only).
All bpy execution happens on the main thread via bpy.app.timers,
which is the only safe way to call bpy.ops from async code in Blender.

Install: Edit → Preferences → Add-ons → Install from Disk
Enable: tick the checkbox, then View3D → N → Agent Bridge → Start Socket Server
"""

bl_info = {
    "name": "Multi-Agent Socket Bridge",
    "author": "blender-multiagent",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Agent Bridge",
    "description": "Thread-safe TCP socket bridge — executes bpy commands on the main thread",
    "category": "Development",
}

import bpy
import socket
import threading
import traceback
import json
import queue


HOST = "127.0.0.1"
PORT = 9876
BUFFER = 65536
EXEC_TIMEOUT = 30.0   # seconds to wait for main-thread execution
TIMER_INTERVAL = 0.05  # seconds between queue polls (50 ms)

_server_thread: threading.Thread | None = None
_server_socket: socket.socket | None = None
_running = False

# Commands waiting to be executed on the main thread.
# Each item: (code_str, threading.Event, list[result_dict])
_queue: queue.Queue = queue.Queue()


# ── Main-thread execution (called by bpy.app.timers) ──────────────────────────

def _process_queue() -> float | None:
    """
    Drain the command queue. Runs on Blender's main thread.
    Returns the next interval, or None to unregister the timer.
    """
    if not _running and _queue.empty():
        return None  # unregister timer when server stops and queue is drained

    while not _queue.empty():
        try:
            code, event, result_box = _queue.get_nowait()
        except queue.Empty:
            break

        local_ns = {"bpy": bpy, "result": None}
        try:
            exec(compile(code, "<agent>", "exec"), local_ns)
            value = local_ns.get("result")
            result_box.append({"status": "ok", "result": str(value) if value is not None else ""})
        except Exception:
            result_box.append({"status": "error", "error": traceback.format_exc()})
        finally:
            event.set()

    return TIMER_INTERVAL


# ── Socket server (background thread — I/O only, no bpy calls) ───────────────

def _handle_client(conn: socket.socket):
    buf = b""
    try:
        while True:
            chunk = conn.recv(BUFFER)
            if not chunk:
                break
            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    msg = {"code": line.decode("utf-8")}

                code = msg.get("code", "")

                # Hand off to the main thread and wait for the result.
                event = threading.Event()
                result_box: list = []
                _queue.put((code, event, result_box))

                if not event.wait(timeout=EXEC_TIMEOUT):
                    response = {"status": "error", "error": "Timed out waiting for main-thread execution"}
                else:
                    response = result_box[0] if result_box else {"status": "error", "error": "No result"}

                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
    except Exception:
        pass
    finally:
        conn.close()


def _server_loop():
    global _server_socket
    _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_socket.bind((HOST, PORT))
    _server_socket.listen(5)
    _server_socket.settimeout(1.0)
    print(f"[Agent Bridge] Listening on {HOST}:{PORT}")

    while _running:
        try:
            conn, _ = _server_socket.accept()
            threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except OSError:
            break

    print("[Agent Bridge] Server stopped.")


# ── Operators ─────────────────────────────────────────────────────────────────

class AGENT_OT_start_server(bpy.types.Operator):
    bl_idname = "agent.start_server"
    bl_label = "Start Socket Server"

    def execute(self, context):
        global _server_thread, _running
        if _running:
            self.report({"WARNING"}, "Server already running")
            return {"CANCELLED"}

        _running = True

        # Register the main-thread timer that drains the command queue.
        if not bpy.app.timers.is_registered(_process_queue):
            bpy.app.timers.register(_process_queue, first_interval=TIMER_INTERVAL, persistent=True)

        _server_thread = threading.Thread(target=_server_loop, daemon=True)
        _server_thread.start()

        self.report({"INFO"}, f"Agent bridge started on port {PORT}")
        return {"FINISHED"}


class AGENT_OT_stop_server(bpy.types.Operator):
    bl_idname = "agent.stop_server"
    bl_label = "Stop Socket Server"

    def execute(self, context):
        global _running, _server_socket
        _running = False
        if _server_socket:
            try:
                _server_socket.close()
            except Exception:
                pass
        self.report({"INFO"}, "Agent bridge stopped")
        return {"FINISHED"}


# ── UI Panel ──────────────────────────────────────────────────────────────────

class AGENT_PT_panel(bpy.types.Panel):
    bl_label = "Agent Bridge"
    bl_idname = "AGENT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Agent Bridge"

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Status: {'Running' if _running else 'Stopped'}")
        layout.label(text=f"Port: {PORT}")
        layout.operator("agent.start_server", icon="PLAY")
        layout.operator("agent.stop_server", icon="PAUSE")


# ── Register ──────────────────────────────────────────────────────────────────

classes = [AGENT_OT_start_server, AGENT_OT_stop_server, AGENT_PT_panel]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    global _running
    _running = False
    if bpy.app.timers.is_registered(_process_queue):
        bpy.app.timers.unregister(_process_queue)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
