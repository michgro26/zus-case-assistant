import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.rag_pipeline import ZUSCaseAssistant


def get_assistant() -> ZUSCaseAssistant:
    root = Path(__file__).resolve().parents[1]
    return ZUSCaseAssistant(root / "data")


def test_detects_dzialalnosc_case():
    assistant = get_assistant()
    result = assistant.analyze_case(
        "Prowadzę działalność i mam e-ZLA. Co jeszcze muszę złożyć do zasiłku chorobowego?",
        ["e-ZLA"],
    )
    assert result["service_type"] == "zasilek_chorobowy_dzialalnosc"
    assert "Z-3b lub ZAS-53" in result["missing_documents"]


def test_detects_ezus_login_case():
    assistant = get_assistant()
    result = assistant.analyze_case(
        "Nie pamiętam loginu do eZUS, ale mam profil zaufany i konto jest zablokowane."
    )
    assert result["service_type"] == "e_zus_logowanie"
    assert "profil zaufany" in result["draft_response"].lower() or "login" in result["draft_response"].lower()