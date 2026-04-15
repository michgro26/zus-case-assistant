# Asystent analizy spraw i korespondencji ZUS

Projekt portfolio pokazujący, jak zbudować narzędzie wspierające pracę na sprawach klientów w środowisku podobnym do ZUS. To nie jest już prosty chatbot-demo. Aplikacja realizuje konkretny workflow:
- analizuje opis sprawy,
- rozpoznaje typ sprawy,
- porównuje go z zanonimizowanymi scenariuszami,
- wykrywa brakujące dokumenty lub elementy,
- wskazuje rekomendowany kanał obsługi,
- przygotowuje roboczy projekt odpowiedzi,
- pokazuje publiczne źródła, na których oparto analizę.

## Charakter projektu
Projekt jest oparty na:
- publicznych materiałach ZUS,
- zanonimizowanych scenariuszach spraw,
- lokalnym pipeline retrieval + rules,
- opcjonalnej redakcji odpowiedzi przez model LLM.

To narzędzie **pomocnicze**. Wynik wymaga weryfikacji w aktualnych procedurach i nie zastępuje oficjalnej interpretacji przepisów ani instrukcji wewnętrznych.

## Co odróżnia ten projekt od wersji demo
- ma workflow „analiza sprawy”, a nie tylko pole pytania,
- pracuje na materiałach publicznych ZUS zamiast generowanych plików testowych,
- rozpoznaje kilka typów spraw,
- wykrywa brakujące dokumenty w prostym, audytowalnym module reguł,
- posiada zanonimizowane scenariusze, które można wykorzystać do testów i prezentacji.

## Moduły aplikacji
### 1. Analiza sprawy
Użytkownik wkleja opis sprawy albo wybiera scenariusz testowy. System zwraca:
- rozpoznany temat,
- dopasowanie do scenariusza,
- brakujące dokumenty,
- rekomendowany kanał,
- checklistę dla pracownika,
- roboczy projekt odpowiedzi,
- źródła publiczne.

### 2. Wyszukiwarka wiedzy
Służy do zadawania pytań o materiały publiczne ZUS, np. o e-wizyty, logowanie do eZUS, zasiłek chorobowy lub zasady kontaktu.

### 3. Materiały źródłowe
Pozwala przejrzeć bazę wiedzy używaną przez system.

## Zakres wiedzy w wersji portfolio
W obecnej wersji projekt obejmuje:
- rezerwację e-wizyty,
- wymagania techniczne e-wizyty,
- kontakt z CKK i bezpieczeństwo danych,
- logowanie do eZUS,
- dokumenty do zasiłku chorobowego.

## Publiczne materiały wykorzystane do przygotowania bazy wiedzy
- ZUS: „Jak zarezerwować e-wizytę”
- ZUS: „E-wizyta - FAQ”
- ZUS: „Sprawdź przed e-wizytą”
- ZUS: „Centrum Kontaktu Klientów (CKK)”
- ZUS: informacja o bezpieczeństwie kontaktu mailowego
- ZUS: „Rejestracja, logowanie i ustawienia konta”
- ZUS: „Niezbędne dokumenty - zasiłek chorobowy z ubezpieczenia chorobowego”

## Architektura
- **UI**: Streamlit
- **Retrieval**: TF-IDF + cosine similarity
- **Scenariusze**: dopasowanie do zanonimizowanych przykładów
- **Decyzje operacyjne**: prosty moduł reguł dla check-list i braków dokumentów
- **LLM**: opcjonalny, tylko do redakcji odpowiedzi na podstawie źródeł

## Struktura repozytorium
```text
zus_case_assistant/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── 01_e_wizyta_rezerwacja.md
│   ├── 02_e_wizyta_wymagania_techniczne.md
│   ├── 03_ckk_i_bezpieczenstwo_kontaktu.md
│   ├── 04_ezus_logowanie.md
│   ├── 05_zasilek_chorobowy_dokumenty.md
│   └── 06_scenariusze_anonimizowane.json
├── src/
│   └── rag_pipeline.py
├── eval/
│   ├── eval_dataset.json
│   └── run_eval.py
└── tests/
    └── test_pipeline.py
```

## Uruchomienie
### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Konfiguracja OpenAI
Aplikacja działa bez klucza API. Jeżeli chcesz, aby system redagował bardziej naturalne odpowiedzi, utwórz plik `.env`:

```env
OPENAI_API_KEY=twoj_klucz
OPENAI_MODEL=gpt-4.1-mini
```

## Przykładowe scenariusze testowe
- klient chce umówić e-wizytę w sprawie zasiłku,
- klient nie pamięta loginu do eZUS,
- przedsiębiorca z e-ZLA pyta o dalsze dokumenty do zasiłku chorobowego,
- klient chce przesłać dane wrażliwe zwykłym mailem i oczekuje pełnej informacji o sprawie.

## Ewaluacja
Uruchom:
```bash
python eval/run_eval.py
```

Skrypt sprawdza m.in.:
- czy system poprawnie rozpoznaje typ sprawy,
- czy wskazuje brakujące dokumenty tam, gdzie powinien,
- czy retrieval zwraca właściwe źródła.

## Jak mówić o tym projekcie na rozmowie
Możesz opisać go tak:

> Zbudowałem narzędzie AI wspierające analizę spraw i korespondencji w środowisku podobnym do ZUS. System korzysta z publicznych materiałów ZUS i zanonimizowanych scenariuszy, rozpoznaje temat sprawy, wskazuje brakujące elementy, rekomenduje kanał obsługi oraz przygotowuje roboczą odpowiedź z podaniem źródeł.

## Ograniczenia
- baza wiedzy obejmuje tylko wybrany zakres spraw,
- moduł brakujących dokumentów opiera się na prostych regułach,
- system nie ma dostępu do danych wewnętrznych ani indywidualnych akt spraw,
- wynik ma charakter pomocniczy.

## Dalszy rozwój
Kierunki rozbudowy:
- embeddings i baza wektorowa,
- lepszy classifier typu sprawy,
- upload PDF i ekstrakcja treści,
- dziennik decyzji i audyt odpowiedzi,
- wersja API w FastAPI,
- dashboard jakości odpowiedzi.
