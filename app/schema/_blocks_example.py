import json
from pathlib import Path

blocks = json.loads(Path(__file__).with_name("zones.geojson").read_text(encoding="utf-8"))
