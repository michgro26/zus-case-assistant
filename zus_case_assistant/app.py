import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import ZUSCaseAssistant

load_dotenv()

st.set_page_config(page_title="Asystent analizy spraw ZUS", page_icon="📄", layout="wide")

DATA_DIR = Path("data")

@st.cache_resource
def get_assistant() -> ZUSCaseAssistant:
    return ZUSCaseAssistant(DATA_DIR)

assistant = get_assistant()

st.title("Asystent analizy spraw i korespondencji")
st.caption("Wersja portfolio oparta na publicznych materiałach ZUS i zanonimizowanych scenariuszach")

with st.sidebar:
    st.header("Konfiguracja")
    use_llm = st.checkbox("Użyj OpenAI do redakcji odpowiedzi", value=bool(os.getenv("OPENAI_API_KEY")))
    top_k = st.slider("Liczba źródeł", 2, 6, 4)
    st.markdown("---")
    st.subheader("Baza wiedzy")
    for name in assistant.document_names:
        st.write(f"- {name}")
    st.markdown("---")
    st.info(
        "Projekt pracuje na publicznych informacjach ZUS. Wynik ma charakter pomocniczy i nie zastępuje weryfikacji w aktualnych procedurach."
    )

case_tab, search_tab, knowledge_tab = st.tabs(["Analiza sprawy", "Wyszukiwarka wiedzy", "Materiały źródłowe"])

with case_tab:
    col1, col2 = st.columns([1.35, 1])
    with col1:
        st.subheader("Opis sprawy")
        selected_scenario = st.selectbox(
            "Wczytaj zanonimizowany scenariusz albo wpisz własny opis",
            options=["Własny opis"] + [f"{s['id']} — {s['title']}" for s in assistant.scenarios],
        )

        scenario_map = {f"{s['id']} — {s['title']}": s for s in assistant.scenarios}
        default_text = ""
        default_docs = []
        if selected_scenario != "Własny opis":
            default_text = scenario_map[selected_scenario]["case_text"]
            default_docs = scenario_map[selected_scenario]["attached_documents"]

        case_text = st.text_area(
            "Treść sprawy / treść pisma klienta",
            value=default_text,
            height=220,
            placeholder="Opisz sprawę, pytanie klienta i aktualny stan dokumentów...",
        )
        attached_documents = st.multiselect(
            "Dokumenty lub informacje, które już są w sprawie",
            options=["e-ZLA", "Z-3", "Z-3b", "ZAS-53", "Z-10", "adres e-mail", "kod pocztowy", "profil zaufany", "PIN do CKK"],
            default=default_docs,
        )
        analyze = st.button("Analizuj sprawę", type="primary")

    with col2:
        st.subheader("Co ocenia system")
        st.write("- rozpoznanie typu sprawy")
        st.write("- dopasowanie do zanonimizowanego scenariusza")
        st.write("- wykrycie brakujących dokumentów lub elementów")
        st.write("- rekomendowany kanał dalszej obsługi")
        st.write("- roboczy projekt odpowiedzi")

    if analyze and case_text.strip():
        with st.spinner("Analizuję sprawę i wyszukuję właściwe źródła..."):
            result = assistant.analyze_case(
                case_text=case_text,
                attached_documents=attached_documents,
                use_llm=use_llm,
                top_k=top_k,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Rozpoznany temat", result["service_label"])
        c2.metric("Dopasowanie scenariusza", f"{result['scenario_similarity']:.2f}")
        c3.metric("Tryb", result["mode"])

        st.subheader("Wynik analizy")
        st.write(f"**Najbliższy scenariusz:** {result['matched_scenario']['title']}")
        st.write(f"**Rekomendowany kanał:** {result['recommended_channel']}")

        if result["missing_documents"]:
            st.warning("Brakujące elementy: " + ", ".join(result["missing_documents"]))
        else:
            st.success("Nie wykryto oczywistych braków dokumentów dla rozpoznanego typu sprawy.")

        st.subheader("Checklista dla pracownika")
        for idx, step in enumerate(result["checklist"], start=1):
            st.write(f"{idx}. {step}")

        st.subheader("Roboczy projekt odpowiedzi")
        st.text_area("Możesz skopiować i dopracować ten szkic", value=result["draft_response"], height=260)

        st.subheader("Źródła publiczne wykorzystane w analizie")
        for idx, src in enumerate(result["knowledge_sources"], start=1):
            with st.expander(f"{idx}. {src['document']} — score={src['score']:.3f}"):
                st.write(f"**Źródło publiczne:** {src['source']}")
                st.write(src["text"])
    elif analyze:
        st.error("Najpierw wpisz opis sprawy.")

with search_tab:
    st.subheader("Pytanie do bazy wiedzy")
    question = st.text_input(
        "Wpisz pytanie o publiczne informacje ZUS",
        placeholder="Np. Jakie dokumenty są potrzebne do zasiłku chorobowego przy działalności gospodarczej?",
    )
    examples = [
        "Jakie dane są potrzebne do rezerwacji e-wizyty?",
        "Jak klient może odzyskać login do eZUS?",
        "Jakie dokumenty są potrzebne do zasiłku chorobowego przy działalności?",
        "Czy klient może przesłać dane wrażliwe zwykłym mailem?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex):
            question = ex

    if question:
        result = assistant.answer_question(question, use_llm=use_llm, top_k=top_k)
        st.subheader("Odpowiedź")
        st.write(result["answer"])
        st.caption(result["mode"])
        st.subheader("Źródła")
        for idx, src in enumerate(result["sources"], start=1):
            with st.expander(f"{idx}. {src['document']} — score={src['score']:.3f}"):
                st.write(f"**Źródło publiczne:** {src['source']}")
                st.write(src["text"])

with knowledge_tab:
    st.subheader("Przegląd materiałów źródłowych")
    selected_doc = st.selectbox("Wybierz dokument", assistant.document_names)
    doc = assistant.knowledge_docs[selected_doc]
    st.write(f"**Źródło publiczne:** {doc['source']}")
    st.text_area("Treść robocza bazy wiedzy", value=doc["text"], height=420)
