# blender-multiagent

Multi-agent AI pipeline that controls Blender 5 via natural language. Type a prompt and three Claude agents collaborate to build, light, and QA-check a 3D scene — with a retry loop if the score is too low.

## How it works

```
Prompt → Orchestrator → Composition Agent → Lighting Agent → QA Agent
                                                                  ↓
                                               score < 7 → retry with feedback
                                               score ≥ 7 → done ✓
```

1. **Orchestrator** — parses the prompt into a structured scene spec (mood, objects, camera, lighting style)
2. **Composition Agent** — generates bpy Python commands to build geometry and position the camera
3. **Lighting Agent** — sets up lights and world background to match the mood
4. **QA Agent** — inspects the scene (JSON snapshot + viewport screenshot) and scores it 1–10. If score < 7 it sends feedback back to the Orchestrator and the whole pipeline retries.

All agents use Claude Sonnet 4.6. Commands are sent to Blender via **BlenderMCP** over stdio → TCP.

## Stack

- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [BlenderMCP](https://github.com/ahujasid/blender-mcp) — MCP server that controls Blender
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- Blender 5 (EEVEE Next renderer)

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install anthropic[mcp] python-dotenv mcp[cli]
brew install uv   # for uvx blender-mcp
```

### 2. API key

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
```

### 3. Install BlenderMCP addon in Blender

1. Download [addon.py](https://github.com/ahujasid/blender-mcp/blob/main/addon.py) from the BlenderMCP repo
2. In Blender: **Edit → Preferences → Add-ons → Install...** → select the file → enable it
3. In the 3D Viewport press **N** → **BlenderMCP** tab → click **Connect**

### 4. Run the pipeline

```bash
source venv/bin/activate
python main.py "a moody cyberpunk street scene at night"
```

## MCP Server

The pipeline is also exposed as an MCP server so any MCP client (Claude Desktop, other agents) can call it as tools.

```bash
source venv/bin/activate
mcp dev mcp_server.py   # opens MCP Inspector in browser
```

**Tools exposed:**

| Tool | Description |
|---|---|
| `build_scene(prompt)` | Run the full pipeline from a prompt |
| `get_scene_screenshot()` | Viewport screenshot of the current scene |
| `run_blender_code(code)` | Execute raw bpy Python in Blender |
| `get_scene_info()` | JSON snapshot of all scene objects |
| `adjust_camera(tilt, z)` | Tweak camera angle without re-running |

### Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blender-multiagent": {
      "command": "/path/to/blender-multiagent/venv/bin/python",
      "args": ["/path/to/blender-multiagent/mcp_server.py"]
    }
  }
}
```

## Project structure

```
blender-multiagent/
├── main.py                  # pipeline entry point
├── blender_bridge.py        # async MCP client (connects to BlenderMCP)
├── mcp_server.py            # exposes pipeline as MCP tools
├── agents/
│   ├── orchestrator.py      # prompt → structured scene spec
│   ├── composition_agent.py # scene spec → bpy geometry commands
│   ├── lighting_agent.py    # scene spec → bpy lighting commands
│   └── qa_agent.py          # scene snapshot → score + fixes
└── pose_fix.py              # one-off character lighting utility
```

## Notes

- Blender must be open with the BlenderMCP addon connected before running
- Character FBX path is hardcoded in `composition_agent.py` — update `CHARACTER_FBX` to your file
- The QA agent uses a viewport screenshot for visual review (requires BlenderMCP connected)
- Renders with EEVEE Next (Blender 5). Final screenshot saved to `/tmp/blender_final_*.png`
