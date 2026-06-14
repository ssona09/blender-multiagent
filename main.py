"""
Entry point — runs the full multi-agent Blender pipeline.

Usage:
    python main.py "a moody cyberpunk street scene at night"
    python main.py          # uses demo prompt
"""

import sys
import os
import json
from dotenv import load_dotenv
import anthropic

from blender_bridge import BlenderBridge, BlenderBridgeError
from agents import orchestrator, composition_agent, lighting_agent, qa_agent

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


def _send_commands(bridge: BlenderBridge, commands: list[str], label: str):
    """Send a list of bpy code strings to Blender, printing each one."""
    print(f"\n[{label}] Executing {len(commands)} command(s)...")
    errors = 0
    for i, cmd in enumerate(commands, 1):
        preview = cmd.split("\n")[0][:80]
        print(f"  [{i}] {preview}{'...' if len(cmd) > 80 else ''}")
        try:
            result = bridge.run(cmd)
            if result and result != "done":
                print(f"       → {result[:400]}")
        except BlenderBridgeError as e:
            errors += 1
            print(f"       ✗ Error:\n{str(e)}")
        except (BrokenPipeError, ConnectionResetError):
            print("       ✗ Blender socket closed mid-run — attempting reconnect...")
            try:
                bridge.reconnect()
                bridge.run(cmd)
                print("       ↻ Retried successfully after reconnect")
            except BlenderBridgeError as e:
                errors += 1
                print(f"       ✗ Reconnect failed: {e}")
    if errors:
        print(f"\n  [{label}] {errors}/{len(commands)} command(s) had errors (continuing)")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(prompt: str):
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    print("\n" + "═" * 55)
    print("  Blender Multi-Agent Pipeline")
    print(f"  Prompt: {prompt}")
    print("═" * 55)

    qa_feedback = ""

    with BlenderBridge() as bridge:
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
            comp_result = composition_agent.run(scene_spec, client)
            _send_commands(bridge, comp_result.get("bpy_commands", []), "Composition")

            comp_summary = comp_result.get("summary", {})
            print(f"\n  Objects placed: {comp_summary.get('objects_placed', [])}")

            # ── 3. Lighting Agent ─────────────────────────────────────
            _log("Lighting Agent", "Setting up lights + world...")
            light_result = lighting_agent.run(scene_spec, comp_summary, client)
            _send_commands(bridge, light_result.get("bpy_commands", []), "Lighting")

            light_summary = light_result.get("summary", {})
            print(f"\n  Mood: {light_summary.get('mood_achieved', '')}")

            # ── 4. QA Agent ───────────────────────────────────────────
            _log("QA Agent", "Inspecting scene...")
            snapshot_json = bridge.run(qa_agent.get_snapshot_code())

            qa_result = qa_agent.run(
                original_prompt=prompt,
                scene_spec=scene_spec,
                scene_snapshot=snapshot_json,
                client=client,
                iteration=attempt,
            )

            score = qa_result.get("score", 0)
            verdict = qa_result.get("verdict", "fail")

            print(f"\n  Score:    {score}/10  ({verdict.upper()})")
            print(f"  Issues:   {qa_result.get('issues', [])}")
            print(f"  Strengths:{qa_result.get('strengths', [])}")

            # Apply any immediate small fixes
            fixes = qa_result.get("immediate_fixes", [])
            if fixes:
                _send_commands(bridge, fixes, "QA Immediate Fixes")

            if verdict == "pass" or score >= PASS_SCORE:
                print(f"\n  ✓ Scene passed QA with score {score}/10!")
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
    user_prompt = " ".join(sys.argv[1:]).strip() or DEMO_PROMPT
    run(user_prompt)
