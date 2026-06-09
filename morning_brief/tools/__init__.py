import json
from pathlib import Path

TOOLS_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tools"
TOOLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_tool_result(tool_name: str, data: dict) -> str:
    path = TOOLS_OUTPUT_DIR / f"{tool_name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)
