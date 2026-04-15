import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
from pathlib import Path

from src.rag_pipeline import ZUSCaseAssistant

ROOT = Path(__file__).resolve().parents[1]
assistant = ZUSCaseAssistant(ROOT / "data")
dataset = json.loads((ROOT / "eval" / "eval_dataset.json").read_text(encoding="utf-8"))

service_ok = 0
missing_ok = 0
source_ok = 0

for item in dataset:
    result = assistant.analyze_case(item["case_text"], item.get("attached_documents", []))
    if result["service_type"] == item["expected_service_type"]:
        service_ok += 1
    if set(item["expected_missing"]).issubset(set(result["missing_documents"])):
        missing_ok += 1
    if any(item["expected_source_contains"] in src["document"] for src in result["knowledge_sources"]):
        source_ok += 1

n = len(dataset)
print({
    "service_accuracy": round(service_ok / n, 2),
    "missing_detection_accuracy": round(missing_ok / n, 2),
    "source_recall": round(source_ok / n, 2),
})