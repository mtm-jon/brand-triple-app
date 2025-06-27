import os, json
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import openai
import streamlit.components.v1 as components

# ── environment ───────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Brand Semantic Triple Generator", layout="wide")
st.title("Brand Semantic Triple Generator")
st.markdown("**Subject | Predicate | Object | Category**")

# ── session defaults ──────────────────────────────────────────────────────
st.session_state.setdefault("synonyms", {})
st.session_state.setdefault("last_df", pd.DataFrame())

# ── user inputs ───────────────────────────────────────────────────────────
brand = st.text_input("Brand (used as Subject in every triple)")

c1, c2, c3, c4 = st.columns(4)
services = c1.text_area("Services / Products")           # updated label
audience = c2.text_area("Audience")
values   = c3.text_area("Value Propositions")
diffs    = c4.text_area("Differentiators")

num_triples = st.slider("Number of triples to generate", 10, 200, 50, 10)
include_category = st.checkbox("Include “category” column", value=True)

# ── helpers ───────────────────────────────────────────────────────────────
def gpt_json(prompt: str):
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def normalize_triples(raw):
    if isinstance(raw, list):
        triples = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("triples"), list):
            triples = raw["triples"]
        elif all(k.isdigit() for k in raw.keys()):
            triples = list(raw.values())
        else:
            triples = [raw]
    else:
        triples = []
    df = pd.DataFrame(triples)
    for col in ("subject", "predicate", "object", "category"):
        if col not in df:
            df[col] = ""
    if not include_category and "category" in df:
        df = df.drop(columns="category")
    return df

def fetch_synonyms(text: str, label: str):
    if not text.strip():
        return []
    prompt = (
        f"For the following {label} terms, suggest 5–10 closely related words "
        f"or phrases. Return ONLY a JSON object with key `synonyms`.\n\n{text}"
    )
    raw = gpt_json(prompt)
    if isinstance(raw, dict) and isinstance(raw.get("synonyms"), list):
        return [w.strip() for w in raw["synonyms"] if isinstance(w, str)]
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v]
    if isinstance(raw, list):
        return [w.strip() for w in raw if isinstance(w, str)]
    return []

COLOR_MAP = {
    "services / products": "#1f77b4",     # updated key
    "services": "#1f77b4",                # keep old for GPT default
    "audience": "#2ca02c",
    "value-propositions": "#ff7f0e",
    "differentiators": "#9467bd",
}

def badge(df: pd.DataFrame):
    if "category" not in df:
        return df
    def style(v):
        return f"background-color:{COLOR_MAP.get(str(v).lower(), '#999')};color:white"
    return df.style.apply(
        lambda col: [style(v) if col.name == "category" else "" for v in col],
        axis=0,
    )

# ── preview one per category ──────────────────────────────────────────────
if st.button("🔎 Preview – one per category"):
    if not brand:
        st.warning("Enter a brand first.")
    else:
        prompt = (
            f'Subject is "{brand}". Generate ONE triple each for '
            "services / products, audience, value-propositions, differentiators. "
            "Return JSON array with subject, predicate, object, category.\n\n"
            f"Services / products: {services}\nAudience: {audience}\n"
            f"Value propositions: {values}\nDifferentiators: {diffs}"
        )
        df_prev = normalize_triples(gpt_json(prompt))
        st.session_state["last_df"] = df_prev
        st.subheader("Preview")
        st.write(badge(df_prev))

# ── generate full set ─────────────────────────────────────────────────────
if st.button(f"⚙️ Generate {num_triples} triples"):
    if not brand:
        st.warning("Enter a brand first.")
    else:
        prompt = (
            f'Subject is "{brand}". Produce EXACTLY {num_triples} triples, '
            "evenly across services / products, audience, value-propositions, "
            "differentiators. Return JSON array with subject, predicate, object, category.\n\n"
            f"Services / products: {services}\nAudience: {audience}\n"
            f"Value propositions: {values}\nDifferentiators: {diffs}"
        )
        df = normalize_triples(gpt_json(prompt))
        st.session_state["last_df"] = df
        st.success(f"{len(df)} triples ready.")
        st.write(badge(df))

# ── clipboard + CSV download ──────────────────────────────────────────────
if not st.session_state["last_df"].empty:
    csv_text = st.session_state["last_df"].to_csv(index=False)
    components.html(
        f"""
        <textarea id="clip" style="opacity:0">{csv_text}</textarea>
        <button onclick="navigator.clipboard.writeText(document.getElementById('clip').value)"
                style="padding:6px 12px;margin:4px 0;">📋 Copy to clipboard</button>
        """,
        height=45,
    )
    st.download_button(
        "💾 Download CSV",
        data=csv_text.encode(),
        file_name=f"{brand.lower().replace(' ','_')}_semantic_triples.csv",
        mime="text/csv",
    )

# ── synonyms helper ───────────────────────────────────────────────────────
if st.button("💡 Suggest similar words"):
    st.session_state["synonyms"] = {
        "Services / Products": fetch_synonyms(services, "service or product"),
        "Audience": fetch_synonyms(audience, "audience"),
        "Value Propositions": fetch_synonyms(values, "value proposition"),
        "Differentiators": fetch_synonyms(diffs, "differentiator"),
    }

if st.session_state["synonyms"]:
    syn_df = (
        pd.DataFrame(st.session_state["synonyms"].items(), columns=["category", "synonyms"])
        .assign(synonyms=lambda d: d.synonyms.apply(", ".join))
    )
    st.subheader("🔄 Suggested Similar Words")
    st.table(syn_df)

# ── footer remains unchanged ──────────────────────────────────────────────
st.markdown(
    """
---  
### 📚 What are Semantic Triples?  
A triple links a *subject* to an *object* through a *predicate*.  
Example: <a href="https://www.movingtrafficmedia.com" target="_blank">Moving Traffic Media</a> (Subject) **offers** (Predicate) *generative engine optimization* (Object).

### 🎯 Why Semantic Triples?  
* Clarity – one fact per sentence  
* Consistency – uniform messaging  
* Machine-readable – ideal for knowledge graphs

### 🤖 Why Semantic Triples for AI / LLMs?  
1. Better retrieval 2. Higher accuracy 3. Stronger entity linking

---

## 🛠️ How to develop triples  

1. Enter brand & attribute examples.  
2. Pick total count and toggle category column.  
3. Preview → Generate → Copy or Download CSV.  
4. Enrich wording with **Suggest similar words**.

---

## 🌍 Where to use them  

| Area | Benefit |
|------|---------|
| **Website & blogs** | Drop sentences directly in copy. |
| **Structured data** | Convert to JSON-LD / RDFa. |
| **Press releases** | Canonical facts for media. |
| **Social captions** | Short, punchy claims. |
| **Knowledge graphs** | Seed chatbots & RAG. |
| **Sales decks** | Slide notes & brochures. |
| **Brand playbooks** | Single source of truth. |
| **Prompt libraries** | Feed brand-safe facts to AI tools. |

---

**Developed by [Jon Clark](https://www.linkedin.com/in/ppcmarketing/)**
""",
    unsafe_allow_html=True,
)
