from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()


@dataclass
class Chunk:
    document: str
    text: str
    source: str


class ZUSCaseAssistant:
    """Local-first assistant for public ZUS knowledge and anonymized case analysis."""

    def __init__(self, data_dir: Path, chunk_size: int = 900, overlap: int = 140) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.knowledge_docs = self._load_markdown_documents()
        self.document_names = sorted(self.knowledge_docs.keys())
        self.scenarios = self._load_scenarios()
        self.chunks = self._build_chunks()

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.chunk_matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

        scenario_texts = [f"{s['title']} {s['case_text']}" for s in self.scenarios]
        self.scenario_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.scenario_matrix = self.scenario_vectorizer.fit_transform(scenario_texts)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if (OpenAI and os.getenv("OPENAI_API_KEY")) else None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        self.service_definitions = {
            "e_wizyta": {
                "label": "Rezerwacja e-wizyty",
                "required_documents": [],
                "keywords": ["e-wizyta", "konsultacja online", "wideorozmowa", "rezerwacja wizyty"],
            },
            "e_zus_logowanie": {
                "label": "Logowanie do eZUS",
                "required_documents": [],
                "keywords": ["ezus", "login", "hasło", "kod weryfikacyjny", "profil zaufany", "blokada konta"],
            },
            "zasilek_chorobowy_pracownik": {
                "label": "Zasiłek chorobowy - pracownik",
                "required_documents": ["e-ZLA", "Z-3"],
                "keywords": ["zasiłek chorobowy", "pracownik", "z-3", "zwolnienie lekarskie"],
            },
            "zasilek_chorobowy_dzialalnosc": {
                "label": "Zasiłek chorobowy - działalność gospodarcza",
                "required_documents": ["e-ZLA", "Z-3b lub ZAS-53"],
                "keywords": ["działalność", "przedsiębiorca", "z-3b", "zas-53", "zasiłek chorobowy"],
            },
            "kontakt_bezpieczenstwo": {
                "label": "Kontakt i bezpieczeństwo danych",
                "required_documents": ["bezpieczny kanał uwierzytelnienia"],
                "keywords": ["mail", "e-mail", "telefon", "pin", "ckk", "dane wrażliwe"],
            },
        }

    def _load_markdown_documents(self) -> Dict[str, Dict[str, str]]:
        docs: Dict[str, Dict[str, str]] = {}
        for path in sorted(self.data_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            source_match = re.search(r"Źródło publiczne:\s*(.+)", raw)
            source = source_match.group(1).strip() if source_match else "brak"
            docs[path.name] = {"text": raw, "source": source}
        if not docs:
            raise ValueError(f"Brak plików .md w {self.data_dir}")
        return docs

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        path = self.data_dir / "06_scenariusze_anonimizowane.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _chunk_text(self, text: str) -> List[str]:
        clean = " ".join(text.split())
        chunks: List[str] = []
        start = 0
        while start < len(clean):
            end = min(start + self.chunk_size, len(clean))
            chunks.append(clean[start:end])
            if end == len(clean):
                break
            start = max(0, end - self.overlap)
        return chunks

    def _build_chunks(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for name, doc in self.knowledge_docs.items():
            for text in self._chunk_text(doc["text"]):
                chunks.append(Chunk(document=name, text=text, source=doc["source"]))
        return chunks

    def retrieve_knowledge(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.chunk_matrix).flatten()
        indices = np.argsort(sims)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in indices:
            results.append(
                {
                    "document": self.chunks[idx].document,
                    "source": self.chunks[idx].source,
                    "text": self.chunks[idx].text,
                    "score": float(sims[idx]),
                }
            )
        return results

    def match_scenario(self, case_text: str) -> Tuple[Dict[str, Any], float]:
        vec = self.scenario_vectorizer.transform([case_text])
        sims = cosine_similarity(vec, self.scenario_matrix).flatten()
        idx = int(np.argmax(sims))
        return self.scenarios[idx], float(sims[idx])

    def detect_service_type(self, case_text: str, scenario_hint: str = "") -> str:
        text = f"{case_text} {scenario_hint}".lower()
        scores: Dict[str, int] = {}
        for key, config in self.service_definitions.items():
            score = sum(1 for kw in config["keywords"] if kw in text)
            scores[key] = score

        if "działal" in text or "przedsiębior" in text:
            scores["zasilek_chorobowy_dzialalnosc"] += 2
        if "pracownik" in text:
            scores["zasilek_chorobowy_pracownik"] += 2
        if "e-zla" in text or "zwolnienie" in text:
            scores["zasilek_chorobowy_pracownik"] += 1
            scores["zasilek_chorobowy_dzialalnosc"] += 1
        if "e-wizyta" in text or "online" in text or "wideo" in text:
            scores["e_wizyta"] += 2
        if "login" in text or "hasł" in text or "profil zaufany" in text:
            scores["e_zus_logowanie"] += 2
        if "mail" in text or "e-mail" in text or "telefon" in text or "pin" in text:
            scores["kontakt_bezpieczenstwo"] += 1

        return max(scores, key=scores.get)

    def detect_missing_documents(self, service_type: str, attached_documents: List[str], case_text: str) -> List[str]:
        required = self.service_definitions.get(service_type, {}).get("required_documents", [])
        normalized_attached = " | ".join(attached_documents + [case_text]).lower()
        missing: List[str] = []
        for doc in required:
            if doc == "bezpieczny kanał uwierzytelnienia":
                if not any(token in normalized_attached for token in ["pin do ckk", "identyfikator ezus", "ezus", "formularz kontaktowy"]):
                    missing.append(doc)
                continue
            tokens = [t.strip().lower() for t in re.split(r"lub|/", doc)]
            if not any(token and token in normalized_attached for token in tokens):
                missing.append(doc)

        if "po ustaniu ubezpieczenia" in case_text.lower() and "z-10" not in normalized_attached:
            missing.append("Z-10")

        return missing

    def recommend_channel(self, service_type: str, case_text: str) -> str:
        text = case_text.lower()
        if service_type == "kontakt_bezpieczenstwo":
            if "stan sprawy" in text or "indywidual" in text:
                return "CKK z uwierzytelnieniem albo konto eZUS"
            return "formularz kontaktowy lub CKK"
        if service_type == "e_wizyta":
            return "rezerwacja e-wizyty przez zus.pl, eZUS, mZUS lub mObywatel"
        if service_type == "e_zus_logowanie":
            return "ekran logowania eZUS / login.gov.pl"
        return "kanał świadczeniowy ZUS zgodny z typem sprawy"

    def build_checklist(self, service_type: str, missing_documents: List[str], case_text: str) -> List[str]:
        checklist: List[str] = []
        if service_type == "e_wizyta":
            checklist = [
                "Potwierdź, jakiego obszaru merytorycznego dotyczy sprawa.",
                "Wskaż kanały rezerwacji: strona ZUS, eZUS, mZUS lub mObywatel.",
                "Przekaż minimalne dane potrzebne do rezerwacji: imię, nazwisko, e-mail i kod pocztowy.",
            ]
        elif service_type == "e_zus_logowanie":
            checklist = [
                "Ustal, czy problem dotyczy loginu, hasła, kodu weryfikacyjnego czy blokady konta.",
                "Sprawdź, czy klient może zalogować się profilem zaufanym lub podpisem kwalifikowanym.",
                "Przekaż ścieżkę odzyskania loginu lub resetu hasła.",
            ]
        elif service_type.startswith("zasilek_chorobowy"):
            checklist = [
                "Zweryfikuj status klienta: pracownik czy działalność gospodarcza.",
                "Sprawdź obecność zwolnienia e-ZLA albo dokumentu równoważnego.",
                "Ustal, czy trzeba dosłać formularz płatnika lub wniosek.",
            ]
        elif service_type == "kontakt_bezpieczenstwo":
            checklist = [
                "Oceń, czy klient próbuje przekazać dane wrażliwe niebezpiecznym kanałem.",
                "Wskaż bezpieczny kanał do spraw indywidualnych.",
                "Wyjaśnij zasady uwierzytelnienia dla pełnej informacji o sprawie.",
            ]

        if missing_documents:
            checklist.append("Wskaż brakujące elementy: " + ", ".join(missing_documents) + ".")

        if "technicz" in case_text.lower() or "kamera" in case_text.lower() or "mikrofon" in case_text.lower():
            checklist.append("Sprawdź przeglądarkę, dostęp do kamery i mikrofonu oraz ewentualne blokady innych aplikacji.")

        return checklist

    def _local_draft(self, analysis: Dict[str, Any]) -> str:
        missing = analysis["missing_documents"]
        sources = analysis["knowledge_sources"]
        lead = f"Rozpoznany temat sprawy: {analysis['service_label']}."

        body_parts = []
        if analysis["matched_scenario"]:
            body_parts.append(
                f"Sprawa jest podobna do scenariusza: {analysis['matched_scenario']['title']} (podobieństwo {analysis['scenario_similarity']:.2f})."
            )
        if missing:
            body_parts.append("Na obecnym etapie widać braki: " + ", ".join(missing) + ".")
        else:
            body_parts.append("Na podstawie opisu nie widać oczywistych braków dokumentów wymaganych dla tego typu sprawy.")
        body_parts.append("Rekomendowany kanał dalszej obsługi: " + analysis["recommended_channel"] + ".")
        body_parts.append("Zalecane kolejne kroki: " + " ".join(f"{i+1}) {step}" for i, step in enumerate(analysis["checklist"])) )

        source_list = "; ".join(f"{src['document']}" for src in sources)
        body_parts.append("Podstawa odpowiedzi pochodzi z materiałów publicznych: " + source_list + ".")
        return lead + "\n\n" + "\n".join(body_parts)

    def _llm_draft(self, analysis: Dict[str, Any]) -> str:
        if not self.client:
            return self._local_draft(analysis)

        context = "\n\n".join(
            f"Dokument: {src['document']}\nŹródło: {src['source']}\nTreść: {src['text']}"
            for src in analysis["knowledge_sources"]
        )
        prompt = f"""
Przygotuj roboczą, rzeczową odpowiedź dla pracownika analizującego sprawę klienta ZUS.
Używaj wyłącznie dostarczonych źródeł i wyniku analizy.
Nie twórz nowych wymagań dokumentowych.
Zaznacz, że odpowiedź ma charakter pomocniczy i wymaga weryfikacji w obowiązujących procedurach.

Wynik analizy:
- temat: {analysis['service_label']}
- brakujące elementy: {', '.join(analysis['missing_documents']) if analysis['missing_documents'] else 'brak oczywistych braków'}
- kanał: {analysis['recommended_channel']}
- checklista: {' | '.join(analysis['checklist'])}

Kontekst źródłowy:
{context}
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Jesteś asystentem analizy spraw. Odpowiadasz po polsku, rzeczowo i tylko na podstawie źródeł."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or self._local_draft(analysis)

    def analyze_case(
        self,
        case_text: str,
        attached_documents: List[str] | None = None,
        use_llm: bool = False,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        attached_documents = attached_documents or []
        matched_scenario, similarity = self.match_scenario(case_text)
        service_type = self.detect_service_type(case_text, matched_scenario.get("service_type", ""))
        missing = self.detect_missing_documents(service_type, attached_documents, case_text)
        knowledge_sources = self.retrieve_knowledge(case_text, top_k=top_k)
        checklist = self.build_checklist(service_type, missing, case_text)

        analysis = {
            "service_type": service_type,
            "service_label": self.service_definitions[service_type]["label"],
            "matched_scenario": matched_scenario,
            "scenario_similarity": similarity,
            "missing_documents": missing,
            "recommended_channel": self.recommend_channel(service_type, case_text),
            "checklist": checklist,
            "knowledge_sources": knowledge_sources,
            "attached_documents": attached_documents,
        }
        analysis["draft_response"] = self._llm_draft(analysis) if use_llm else self._local_draft(analysis)
        analysis["mode"] = f"LLM ({self.model}) + retrieval + rules" if use_llm and self.client else "lokalny retrieval + rules"
        return analysis

    def answer_question(self, question: str, use_llm: bool = False, top_k: int = 4) -> Dict[str, Any]:
        sources = self.retrieve_knowledge(question, top_k=top_k)
        answer = self._local_draft(
            {
                "service_label": "Wyszukiwanie wiedzy",
                "matched_scenario": None,
                "scenario_similarity": 0.0,
                "missing_documents": [],
                "recommended_channel": "zależny od treści pytania",
                "checklist": ["Przejrzyj wskazane źródła i zweryfikuj, czy odpowiadają na pytanie."],
                "knowledge_sources": sources,
            }
        )
        if use_llm and self.client:
            context = "\n\n".join(f"Dokument: {s['document']}\nTreść: {s['text']}" for s in sources)
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "Odpowiadasz po polsku, wyłącznie na podstawie materiałów publicznych ZUS i dołączonych źródeł."},
                    {"role": "user", "content": f"Pytanie: {question}\n\nŹródła:\n{context}"},
                ],
            )
            answer = resp.choices[0].message.content or answer
        return {"answer": answer, "sources": sources, "mode": "lokalny retrieval" if not (use_llm and self.client) else f"LLM ({self.model}) + retrieval"}


if __name__ == "__main__":
    assistant = ZUSCaseAssistant(data_dir=Path("data"))
    sample = assistant.analyze_case(
        "Prowadzę działalność i mam e-ZLA. Jakie dokumenty muszę jeszcze złożyć do zasiłku chorobowego?",
        attached_documents=["e-ZLA"],
    )
    print(json.dumps(sample, ensure_ascii=False, indent=2))
