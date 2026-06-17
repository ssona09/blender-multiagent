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

import os
import base64
from dotenv import load_dotenv
import anthropic
from mcp.server.fastmcp import FastMCP, Context, Image

load_dotenv()

from blender_bridge import BlenderBridge, BlenderBridgeError
from agents import orchestrator, composition_agent, lighting_agent, qa_agent

MAX_RETRIES = 2
PASS_SCORE = 7

mcp = FastMCP("blender-multiagent")

_bridge: BlenderBridge | None = None


async def _get_bridge() -> BlenderBridge:
    """Return the persistent bridge, reconnecting if it went stale."""
    global _bridge
    if _bridge is None:
        _bridge = BlenderBridge()
        await _bridge.__aenter__()
    return _bridge


async def _run(code: str) -> str:
    """Run bpy code, reconnecting once if the bridge dropped."""
    global _bridge
    try:
        bridge = await _get_bridge()
        return await bridge.run(code)
    except (BlenderBridgeError, Exception):
        # Bridge went stale — reconnect and retry once
        try:
            if _bridge:
                await _bridge.__aexit__(None, None, None)
        except Exception:
            pass
        _bridge = None
        bridge = await _get_bridge()
        return await bridge.run(code)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def build_scene(prompt: str, ctx: Context) -> str:
    """
    Build a complete 3D Blender scene from a natural language prompt.
    Runs orchestrator → composition → lighting → QA, retrying up to 2x if score < 7.
    Returns a progress log. Call get_scene_screenshot() afterwards to see the result.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ["Error: ANTHROPIC_API_KEY not set"]

    client = anthropic.Anthropic(api_key=api_key)
    bridge = await _get_bridge()
    log = []
    qa_feedback = ""
    score = 0
    total_steps = (MAX_RETRIES + 1) * 4  # 4 stages per attempt

    await ctx.info(f"Starting pipeline for: {prompt}")

    for attempt in range(1, MAX_RETRIES + 2):
        step_base = (attempt - 1) * 4
        log.append(f"\n── Attempt {attempt}/{MAX_RETRIES + 1} ──")

        # Orchestrator
        await ctx.report_progress(step_base, total_steps, "Orchestrator: parsing prompt...")
        scene_spec = orchestrator.run(prompt, client, feedback=qa_feedback)
        log.append(f"Orchestrator: {scene_spec.get('mood', '')} scene, {scene_spec.get('time_of_day', '')}")
        await ctx.info(f"Scene spec: {scene_spec.get('mood')} / {scene_spec.get('style')}")

        # Composition
        await ctx.report_progress(step_base + 1, total_steps, "Composition: building geometry...")
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
        await ctx.info(f"Placed: {objects}")

        # Lighting
        await ctx.report_progress(step_base + 2, total_steps, "Lighting: setting up lights...")
        light_result = lighting_agent.run(scene_spec, comp_result.get("summary", {}), client)
        for cmd in light_result.get("bpy_commands", []):
            try:
                await bridge.run(cmd)
            except BlenderBridgeError as e:
                log.append(f"  Lighting error: {str(e)[:100]}")
        mood = light_result.get("summary", {}).get("mood_achieved", "")
        log.append(f"Lighting: {mood}")
        await ctx.info(f"Mood: {mood}")

        # QA
        await ctx.report_progress(step_base + 3, total_steps, "QA: reviewing scene...")
        snapshot = await bridge.run(qa_agent.get_snapshot_code())
        screenshot = await bridge.get_viewport_screenshot()
        qa_result = qa_agent.run(prompt, scene_spec, snapshot, client, attempt, screenshot)

        score = qa_result.get("score", 0)
        verdict = qa_result.get("verdict", "fail")
        issues = qa_result.get("issues", [])
        log.append(f"QA: {score}/10 — {verdict.upper()}")
        if issues:
            log.append(f"  Issues: {issues}")
        await ctx.info(f"QA score: {score}/10 ({verdict}) — issues: {issues}")

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
            log.append(f"✗ Score {score} < {PASS_SCORE} — retrying with feedback...")
            await ctx.info(f"Retrying — feedback: {qa_feedback[:120]}")
        else:
            log.append(f"✗ Max retries reached. Final score: {score}/10.")

    await ctx.report_progress(total_steps, total_steps, "Done")
    return "\n".join(log)


@mcp.tool()
async def get_scene_screenshot(ctx: Context) -> Image:
    """
    Capture a viewport screenshot of the current Blender scene.
    """
    bridge = await _get_bridge()
    shot = await bridge.get_viewport_screenshot()
    if not shot:
        raise ValueError("No screenshot available — is Blender open with BlenderMCP connected?")
    await ctx.info("Screenshot captured")
    return Image(data=base64.b64decode(shot), format="png")


@mcp.tool()
async def run_blender_code(code: str) -> str:
    """
    Execute arbitrary bpy Python code directly in Blender and return the result.
    Always end your code with: result = "description of what you did"
    """
    try:
        return await _run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def get_scene_info() -> str:
    """
    Get a detailed JSON snapshot of every object in the current Blender scene —
    names, types, locations, scales, light energies, camera focal length.
    """
    try:
        return await _run(qa_agent.get_snapshot_code())
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def clear_scene() -> str:
    """
    Delete all objects from the current Blender scene, leaving a blank canvas.
    Run this before build_scene if you want a fresh start.
    """
    code = """
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
result = f"Scene cleared"
"""
    try:
        return await _run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def adjust_camera(tilt_degrees: float = 88.0, z_height: float | None = None) -> str:
    """
    Adjust the scene camera tilt and optional height.
    tilt_degrees: X rotation in degrees (80=standard, 88=looking more down, 90=straight down)
    z_height: camera Z position in meters — omit to keep current
    """
    z_line = f"cam.location.z = {z_height}" if z_height is not None else "# z unchanged"
    code = f"""
import math, bpy
cam = next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
if not cam:
    result = 'No camera found'
else:
    bpy.context.scene.camera = cam
    cam.rotation_euler[0] = math.radians({tilt_degrees})
    {z_line}
    result = f"Camera: tilt={{math.degrees(cam.rotation_euler[0]):.1f}}deg, z={{cam.location.z:.2f}}"
"""
    try:
        return await _run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


@mcp.tool()
async def adjust_light(name: str, energy: float | None = None, color_hex: str | None = None) -> str:
    """
    Adjust a specific light's energy and/or colour without rebuilding the scene.
    name: exact object name of the light (e.g. 'KeyLight', 'CharFill')
    energy: new energy value in watts (e.g. 800)
    color_hex: hex colour string without # (e.g. 'ff8800' for orange)
    """
    energy_line = f"light.data.energy = {energy}" if energy is not None else "# energy unchanged"

    if color_hex:
        r = int(color_hex[0:2], 16) / 255
        g = int(color_hex[2:4], 16) / 255
        b = int(color_hex[4:6], 16) / 255
        color_line = f"light.data.color = ({r:.4f}, {g:.4f}, {b:.4f})"
    else:
        color_line = "# color unchanged"

    code = f"""
import bpy
light = bpy.data.objects.get({name!r})
if not light or light.type != 'LIGHT':
    result = f"Light {name!r} not found"
else:
    {energy_line}
    {color_line}
    result = f"{{light.name}}: energy={{light.data.energy}}, color={{list(light.data.color)}}"
"""
    try:
        return await _run(code)
    except BlenderBridgeError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
