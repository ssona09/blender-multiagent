"""
Blender Multi-Agent MCP Server

Exposes the full AI pipeline as MCP tools so any MCP client
(Claude Desktop, other agents, etc.) can build Blender scenes via natural language.

Usage — add to Claude Desktop config (~/.claude/claude_desktop_config.json):
{
  "mcpServers": {
    "blender-multiagent": {
      "command": "/Users/sonali/Documents/blender-multiagent/venv/bin/python",
      "args": ["/Users/sonali/Documents/blender-multiagent/mcp_server.py"]
    }
  }
}

Requires: BlenderMCP addon running in Blender (N-panel → BlenderMCP → Connect)
"""

import asyncio
import os
import json
from dotenv import load_dotenv
import anthropic
from mcp.server.fastmcp import FastMCP

load_dotenv()

from blender_bridge import BlenderBridge, BlenderBridgeError
from agents import orchestrator, composition_agent, lighting_agent, qa_agent

MAX_RETRIES = 2
PASS_SCORE = 7

mcp = FastMCP("blender-multiagent")

# Keep one bridge alive for the server lifetime
_bridge: BlenderBridge | None = None
_bridge_stack = None


async def _get_bridge() -> BlenderBridge:
    global _bridge, _bridge_stack
    if _bridge is None:
        _bridge = BlenderBridge()
        await _bridge.__aenter__()
    return _bridge


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def build_scene(prompt: str) -> str:
    """
    Build a complete 3D Blender scene from a natural language prompt.
    Runs the full orchestrator → composition → lighting → QA pipeline.
    Returns a summary of each stage and the final QA score.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)
    bridge = await _get_bridge()
    log = []
    qa_feedback = ""
    score = 0
    qa_result = {}

    for attempt in range(1, MAX_RETRIES + 2):
        log.append(f"\n── Attempt {attempt}/{MAX_RETRIES + 1} ──")

        # Orchestrator
        scene_spec = orchestrator.run(prompt, client, feedback=qa_feedback)
        log.append(f"Orchestrator: {scene_spec.get('mood', '')} scene, {scene_spec.get('time_of_day', '')}")

        # Composition
        comp_result = composition_agent.run(scene_spec, client)
        errors = 0
        for cmd in comp_result.get("bpy_commands", []):
            try:
                await bridge.run(cmd)
            except BlenderBridgeError as e:
                errors += 1
                log.append(f"  Composition error: {str(e)[:100]}")
        objects = comp_result.get("summary", {}).get("objects_placed", [])
        log.append(f"Composition: placed {objects} ({errors} errors)")

        # Lighting
        light_result = lighting_agent.run(scene_spec, comp_result.get("summary", {}), client)
        for cmd in light_result.get("bpy_commands", []):
            try:
                await bridge.run(cmd)
            except BlenderBridgeError as e:
                log.append(f"  Lighting error: {str(e)[:100]}")
        mood = light_result.get("summary", {}).get("mood_achieved", "")
        log.append(f"Lighting: {mood}")

        # QA
        snapshot = await bridge.run(qa_agent.get_snapshot_code())
        screenshot = await bridge.get_viewport_screenshot()
        qa_result = qa_agent.run(prompt, scene_spec, snapshot, client, attempt, screenshot)

        score = qa_result.get("score", 0)
        verdict = qa_result.get("verdict", "fail")
        log.append(f"QA: {score}/10 — {verdict.upper()}")
        log.append(f"  Issues: {qa_result.get('issues', [])}")

        for fix in qa_result.get("immediate_fixes", []):
            try:
                await bridge.run(fix)
            except BlenderBridgeError as e:
                log.append(f"  Fix error: {str(e)[:80]}")

        if verdict == "pass" or score >= PASS_SCORE:
            log.append(f"\n✓ Scene passed QA with score {score}/10!")
            break

        if attempt <= MAX_RETRIES:
            qa_feedback = qa_result.get("feedback_for_orchestrator", "")
            log.append(f"✗ Score {score} < {PASS_SCORE} — retrying...")
        else:
            log.append(f"✗ Max retries reached. Final score: {score}/10.")

    return "\n".join(log)


@mcp.tool()
async def get_scene_screenshot() -> list:
    """
    Capture a viewport screenshot of the current Blender scene.
    Returns the image so you can visually inspect what's in the scene.
    """
    bridge = await _get_bridge()
    shot = await bridge.get_viewport_screenshot()
    if not shot:
        return [{"type": "text", "text": "No screenshot available — is Blender open with BlenderMCP connected?"}]
    return [{"type": "image", "data": shot, "mimeType": "image/png"}]


@mcp.tool()
async def run_blender_code(code: str) -> str:
    """
    Execute arbitrary bpy Python code directly in Blender and return the result.
    Use this for one-off fixes, queries, or custom adjustments to the scene.
    Always end your code with: result = "description of what you did"
    """
    bridge = await _get_bridge()
    try:
        return await bridge.run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def get_scene_info() -> str:
    """
    Get a detailed JSON snapshot of every object in the current Blender scene —
    names, types, locations, scales, light energies, camera focal length.
    """
    bridge = await _get_bridge()
    try:
        return await bridge.run(qa_agent.get_snapshot_code())
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def adjust_camera(tilt_degrees: float = 88.0, z_height: float | None = None) -> str:
    """
    Adjust the scene camera tilt and optional height.
    tilt_degrees: X rotation in degrees (80=standard, 88=looking more down, 90=straight down)
    z_height: camera Z position in meters (leave None to keep current)
    """
    bridge = await _get_bridge()
    code = f"""
import math, bpy
cam = next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
if not cam:
    result = 'No camera found'
else:
    bpy.context.scene.camera = cam
    cam.rotation_euler[0] = math.radians({tilt_degrees})
    {"cam.location.z = " + str(z_height) if z_height is not None else "# z unchanged"}
    result = f"Camera: tilt={{math.degrees(cam.rotation_euler[0]):.1f}}deg, z={{cam.location.z:.2f}}"
"""
    try:
        return await bridge.run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
