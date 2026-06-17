"""
Entry point — runs the full multi-agent Blender pipeline.

Usage:
    python main.py "a moody cyberpunk street scene at night"
    python main.py          # uses demo prompt

Requires:
    - BlenderMCP addon installed and connected in Blender (N-panel → BlenderMCP → Connect)
    - `brew install uv` (for uvx blender-mcp)
    - `pip install "anthropic[mcp]"`
"""

import sys
import os
import json
import asyncio
from dotenv import load_dotenv
import anthropic

from blender_bridge import BlenderBridge, BlenderBridgeError
from agents import orchestrator, composition_agent, spatial_agent, lighting_agent, qa_agent

MAX_RETRIES = 2   # QA retries if score < 7
PASS_SCORE = 7

DEMO_PROMPT = "a dark fantasy forest with dramatic moonlight"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(section: str, msg: str = ""):
    print(f"\n{'─'*55}")
    print(f"  {section}")
    if msg:
        print(f"  {msg}")
    print(f"{'─'*55}")


async def _send_commands(bridge: BlenderBridge, commands: list[str], label: str):
    """Send a list of bpy code strings to Blender, printing each one."""
    print(f"\n[{label}] Executing {len(commands)} command(s)...")
    errors = 0
    for i, cmd in enumerate(commands, 1):
        preview = cmd.split("\n")[0][:80]
        print(f"  [{i}] {preview}{'...' if len(cmd) > 80 else ''}")
        try:
            result = await bridge.run(cmd)
            if result and result != "done":
                print(f"       → {result[:400]}")
        except BlenderBridgeError as e:
            errors += 1
            print(f"       ✗ Error:\n{str(e)}")
    if errors:
        print(f"\n  [{label}] {errors}/{len(commands)} command(s) had errors (continuing)")


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run(prompt: str, include_character: bool = True):
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    print("\n" + "═" * 55)
    print("  Blender Multi-Agent Pipeline  (BlenderMCP)")
    print(f"  Prompt: {prompt}")
    print("═" * 55)

    qa_feedback = ""

    async with BlenderBridge() as bridge:
        for attempt in range(1, MAX_RETRIES + 2):  # +2: initial + retries
            is_retry = attempt > 1
            label = f"Attempt {attempt}/{MAX_RETRIES + 1}"

            # ── 1. Orchestrator ───────────────────────────────────────
            _log(f"Orchestrator Agent  [{label}]",
                 "Retry — incorporating QA feedback" if is_retry else "Parsing prompt → scene spec")
            scene_spec = orchestrator.run(prompt, client, feedback=qa_feedback)
            print(json.dumps(scene_spec, indent=2))

            # ── 2. Composition Agent ──────────────────────────────────
            _log("Composition Agent", "Building geometry + camera...")
            comp_result = composition_agent.run(scene_spec, client, include_character=include_character)
            await _send_commands(bridge, comp_result.get("bpy_commands", []), "Composition")

            comp_summary = comp_result.get("summary", {})
            print(f"\n  Objects placed: {comp_summary.get('objects_placed', [])}")

            # ── 3. Spatial Agent ──────────────────────────────────────
            _log("Spatial Agent", "Checking for floating objects, scale issues, clipping...")
            spatial_snapshot = await bridge.run(qa_agent.get_snapshot_code())
            spatial_shot = await bridge.get_viewport_screenshot()
            spatial_result = spatial_agent.run(scene_spec, spatial_snapshot, client, spatial_shot)

            print(f"\n  Spatial issues: {spatial_result.get('issues', [])}")
            spatial_fixes = spatial_result.get("fixes", [])
            if spatial_fixes:
                await _send_commands(bridge, spatial_fixes, "Spatial Fixes")
            else:
                print("  No spatial issues found.")

            # ── 4. Lighting Agent ─────────────────────────────────────
            _log("Lighting Agent", "Setting up lights + world...")
            light_result = lighting_agent.run(scene_spec, comp_summary, client)
            await _send_commands(bridge, light_result.get("bpy_commands", []), "Lighting")

            light_summary = light_result.get("summary", {})
            print(f"\n  Mood: {light_summary.get('mood_achieved', '')}")

            # ── 5. QA Agent ───────────────────────────────────────────
            _log("QA Agent", "Inspecting scene...")
            snapshot_json = await bridge.run(qa_agent.get_snapshot_code())

            # Grab a viewport screenshot for visual QA
            print("  [QA] Capturing viewport screenshot...")
            screenshot_b64 = await bridge.get_viewport_screenshot()
            if screenshot_b64:
                print("  [QA] Screenshot captured — QA agent can see the scene visually")
            else:
                print("  [QA] No screenshot available — falling back to JSON-only review")

            qa_result = qa_agent.run(
                original_prompt=prompt,
                scene_spec=scene_spec,
                scene_snapshot=snapshot_json,
                client=client,
                iteration=attempt,
                screenshot_b64=screenshot_b64,
            )

            score = qa_result.get("score", 0)
            verdict = qa_result.get("verdict", "fail")

            print(f"\n  Score:    {score}/10  ({verdict.upper()})")
            print(f"  Issues:   {qa_result.get('issues', [])}")
            print(f"  Strengths:{qa_result.get('strengths', [])}")

            # Apply any immediate small fixes
            fixes = qa_result.get("immediate_fixes", [])
            if fixes:
                await _send_commands(bridge, fixes, "QA Immediate Fixes")

            if verdict == "pass" or score >= PASS_SCORE:
                print(f"\n  ✓ Scene passed QA with score {score}/10!")
                # Save a viewport screenshot as the final output
                final_shot = await bridge.get_viewport_screenshot()
                if final_shot:
                    import base64, time
                    out = f"/tmp/blender_final_{int(time.time())}.png"
                    with open(out, "wb") as f:
                        f.write(base64.b64decode(final_shot))
                    print(f"  [Pipeline] Final screenshot saved to: {out}")
                break

            if attempt <= MAX_RETRIES:
                qa_feedback = qa_result.get("feedback_for_orchestrator", "")
                print(f"\n  ✗ Score {score} < {PASS_SCORE} — retrying with feedback...")
                print(f"  Feedback: {qa_feedback[:200]}")
            else:
                print(f"\n  ✗ Max retries reached. Final score: {score}/10.")

    print("\n" + "═" * 55)
    print("  Pipeline complete — scene is live in Blender!")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    no_char = "--no-character" in args
    args = [a for a in args if a != "--no-character"]
    user_prompt = " ".join(args).strip() or DEMO_PROMPT
    asyncio.run(run(user_prompt, include_character=not no_char))
