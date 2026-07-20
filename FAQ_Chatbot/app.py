import streamlit as st
import json
import os
import pandas as pd
import io
from nlp_processor import match_faq_local, match_faq_groq, preprocess_text, get_confidence_tier

# Set page config
st.set_page_config(
    page_title="Aether AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

FAQ_FILE = "faqs.json"

def load_faqs():
    if not os.path.exists(FAQ_FILE):
        return []
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading FAQs: {e}")
        return []

def save_faqs(faqs):
    try:
        with open(FAQ_FILE, "w", encoding="utf-8") as f:
            json.dump(faqs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving FAQs: {e}")

faqs = load_faqs()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am the Aether AI Assistant. Ask me anything about Aether Cloud Platform, deployments, pricing, security, or support!",
            "metadata": None
        }
    ]

if "last_diagnostics" not in st.session_state:
    st.session_state.last_diagnostics = {
        "query": "",
        "tokens": [],
        "scores": [],
        "engine": "N/A"
    }

if "query_log" not in st.session_state:
    st.session_state.query_log = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "editing_idx" not in st.session_state:
    st.session_state.editing_idx = None

# Custom CSS Theme
st.markdown("""
<style>
    /* Styling Main Title */
    .glow-text {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 18px rgba(0, 242, 254, 0.35);
        margin-bottom: 2px;
    }
    .sub-text {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .glass-card {
        background: rgba(17, 25, 40, 0.7);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 15px;
    }
    .section-title {
        color: #00f2fe;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(0, 242, 254, 0.2);
        padding-bottom: 6px;
    }
    .confidence-badge {
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.82rem;
        color: #ffffff;
        display: inline-block;
    }
    /* Streamlit Buttons styling */
    div.stButton > button {
        border-radius: 18px;
        border: 1px solid #4facfe;
        color: #4facfe !important;
        background-color: rgba(0, 242, 254, 0.05);
        transition: all 0.25s ease-in-out;
    }
    div.stButton > button:hover {
        color: white !important;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        border-color: transparent;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #00f2fe;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

engine = st.sidebar.radio(
    "Select Matching Engine",
    ["Hybrid Local Engine (TF-IDF + Fuzzy Ratio)", "Groq LLM Engine (Llama 3.1)"],
    index=0
)

threshold = 0.30
tfidf_weight = 0.65
fuzzy_weight = 0.35

if "Hybrid" in engine:
    st.sidebar.markdown("##### 🎛️ Hybrid Algorithm Tuning")
    threshold = st.sidebar.slider(
        "Match Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Minimum required confidence score to accept an FAQ match."
    )
    tfidf_weight = st.sidebar.slider(
        "TF-IDF Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Weight assigned to keyword TF-IDF cosine similarity."
    )
    fuzzy_weight = round(1.0 - tfidf_weight, 2)
    st.sidebar.caption(f"Fuzzy Match Weight automatically set to: `{fuzzy_weight}`")

st.sidebar.markdown("---")

groq_key_input = st.sidebar.text_input(
    "Groq API Key (Optional Override)",
    type="password",
    placeholder="gsk_...",
    help="Override key loaded from environment `.env` file."
)

st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am the Aether AI Assistant. Ask me anything about Aether Cloud Platform, deployments, pricing, security, or support!",
            "metadata": None
        }
    ]
    st.session_state.last_diagnostics = {
        "query": "",
        "tokens": [],
        "scores": [],
        "engine": "N/A"
    }
    st.rerun()

st.sidebar.markdown(f"**FAQ Database Size:** `{len(faqs)}` entries")
if len(faqs) > 0:
    categories = sorted(list(set([f.get("category", "General") for f in faqs])))
    st.sidebar.markdown(f"**Categories:** {', '.join([f'`{cat}`' for cat in categories])}")

# Header
st.markdown("<h1 class='glow-text'>🤖 Aether AI FAQ Chatbot & Diagnostic Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Intelligent customer support assistant powered by Hybrid NLP (TF-IDF + Levenshtein Fuzzy Matching) & Groq LLM.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 FAQ Chatbot", "📊 Diagnostics & Analytics", "⚙️ FAQ Database Manager"])

# Process query input
user_input = st.session_state.pending_query
st.session_state.pending_query = None

chat_input = st.chat_input("Ask a question about Aether Cloud...")
if chat_input:
    user_input = chat_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "metadata": None})

    if "Hybrid" in engine:
        match, score, scores, tokens = match_faq_local(user_input, faqs, threshold, tfidf_weight, fuzzy_weight)

        conf = get_confidence_tier(score)
        st.session_state.last_diagnostics = {
            "query": user_input,
            "tokens": tokens,
            "scores": scores,
            "engine": "Hybrid Local (TF-IDF + Fuzzy Ratio)",
            "score": score,
            "confidence": conf['level']
        }

        # Log query metric
        st.session_state.query_log.append({
            "query": user_input,
            "matched_question": match["question"] if match else "None",
            "score": score,
            "confidence": conf['level'],
            "engine": "Hybrid Local",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        if match:
            response = match["answer"]
            metadata = {
                "matched_question": match["question"],
                "score": score,
                "confidence": conf['level'],
                "confidence_color": conf['color'],
                "category": match.get("category", "General"),
                "engine": "Hybrid Local NLP"
            }
        else:
            closest = scores[0] if scores else None
            response = "I couldn't find a direct answer with sufficient confidence in our FAQ database for your question."
            if closest:
                response += f"\n\n**Closest Suggested FAQ:** {closest['question']}\n**Answer:** {next((f['answer'] for f in faqs if f['question'] == closest['question']), 'N/A')}\n\n*If you need further help, please contact our support team at support@aether.io.*"
            metadata = {
                "matched_question": closest["question"] if closest else "None",
                "score": closest["score"] if closest else 0.0,
                "confidence": "Low / Unmatched",
                "confidence_color": "#ff5252",
                "category": closest.get("category", "General") if closest else "N/A",
                "engine": "Hybrid Local NLP (Fallback)"
            }

        st.session_state.messages.append({"role": "assistant", "content": response, "metadata": metadata})

    else:
        # Groq engine
        api_key = groq_key_input if groq_key_input else None
        _, _, local_scores, tokens = match_faq_local(user_input, faqs, 0.0)

        with st.spinner("Analyzing semantic intent with Groq LLM..."):
            res = match_faq_groq(user_input, faqs, st.session_state.messages, api_key)

        st.session_state.last_diagnostics = {
            "query": user_input,
            "tokens": tokens,
            "scores": local_scores,
            "engine": "Groq LLM (Llama-3.1-8b-instant)",
            "reasoning": res.get("reasoning", "N/A"),
            "tokens_used": res.get("tokens_used", 0)
        }

        st.session_state.query_log.append({
            "query": user_input,
            "matched_question": res.get("matched_question", "N/A"),
            "score": 1.0 if res.get("matched_index", -1) != -1 else 0.0,
            "confidence": "High" if res.get("matched_index", -1) != -1 else "Low",
            "engine": "Groq LLM",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        response = res.get("answer", "Error retrieving answer.")
        metadata = {
            "matched_question": res.get("matched_question", "N/A"),
            "reasoning": res.get("reasoning", "N/A"),
            "tokens_used": res.get("tokens_used", 0),
            "engine": "Groq LLM"
        }

        st.session_state.messages.append({"role": "assistant", "content": response, "metadata": metadata})

    st.rerun()

# Tab 1: Chatbot
with tab1:
    col_chat, col_sidebar_tools = st.columns([3.2, 1])

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["metadata"]:
                    meta = msg["metadata"]
                    if "Hybrid Local" in meta["engine"]:
                        st.markdown(
                            f"""<div style='font-size: 0.82rem; color: #a0aec0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; margin-top: 6px;'>
                            🎯 <b>Engine:</b> {meta['engine']} | Confidence: <span class='confidence-badge' style='background-color: {meta.get("confidence_color", "#00f2fe")};'>{meta.get("confidence", "N/A")} ({meta['score']:.3f})</span><br/>
                            📌 <b>Matched FAQ:</b> <i>"{meta['matched_question']}"</i>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    elif meta["engine"] == "Groq LLM":
                        st.markdown(
                            f"""<div style='font-size: 0.82rem; color: #a0aec0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; margin-top: 6px;'>
                            ⚡ <b>Groq Semantic Match</b> | Matched Question: <i>"{meta['matched_question']}"</i><br/>
                            🧠 <b>Reasoning:</b> {meta['reasoning']}<br/>
                            📊 <b>Tokens Used:</b> {meta['tokens_used']}
                            </div>""",
                            unsafe_allow_html=True
                        )

    with col_sidebar_tools:
        st.markdown("<p style='font-weight: 600; color: #00f2fe; margin-bottom: 10px;'>💡 Quick Test Questions</p>", unsafe_allow_html=True)
        suggested_queries = [
            "What is Aether Cloud Platform?",
            "How do I sign up for account?",
            "refnd policyyy?", # Typos test
            "Does Aether support multi-region?",
            "Is data encrypted?",
            "How does auto scaling work?",
            "Contact support email"
        ]

        for idx, sq in enumerate(suggested_queries):
            if st.button(sq, key=f"sq_{idx}", use_container_width=True):
                st.session_state.pending_query = sq
                st.rerun()

        st.markdown("---")
        st.markdown("<p style='font-weight: 600; color: #00f2fe; margin-bottom: 10px;'>📥 Export Chat Session</p>", unsafe_allow_html=True)

        # Export JSON
        chat_json_str = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=chat_json_str,
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True
        )

        # Export Markdown
        md_content = "# Aether Chatbot Transcript\n\n"
        for m in st.session_state.messages:
            md_content += f"### {m['role'].capitalize()}\n{m['content']}\n\n"
        st.download_button(
            label="📝 Download Markdown",
            data=md_content,
            file_name="chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

# Tab 2: Diagnostics & Analytics
with tab2:
    st.markdown("<h3 class='section-title'>🔍 Real-Time Diagnostics & Analytics</h3>", unsafe_allow_html=True)

    diag = st.session_state.last_diagnostics
    if not diag["query"]:
        st.info("Submit a question in the chat tab to view real-time vector, token, and confidence metrics.")
    else:
        col_preprocess, col_matching = st.columns([1, 1])

        with col_preprocess:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>1. Text Preprocessing Pipeline</h4>", unsafe_allow_html=True)
            st.markdown(f"**Raw User Query:** `{diag['query']}`")

            st.markdown("**Processed Tokens (NLTK Tokenized + Lowercased + Stopwords Removed + Lemmatized):**")
            if diag["tokens"]:
                badges_html = "".join([f"<span style='background-color:#1a202c; color:#00f2fe; border: 1px solid #00f2fe; padding:4px 10px; margin:3px; border-radius:12px; display:inline-block; font-size:0.85rem;'>{t}</span>" for t in diag["tokens"]])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.markdown("*No tokens remained after stopword filtering.*")

            st.markdown("---")
            st.markdown("##### NLP Preprocessing Stages:")
            st.markdown("""
            1. **Lowercased & Normalization**: Standardizes casing for uniform lookup.
            2. **Tokenization**: NLTK `word_tokenize` breaks sentence into discrete word units.
            3. **Punctuation Stripping**: Filters out special characters.
            4. **Stopword Removal**: Removes high-frequency auxiliary words (*is*, *the*, *for*, *what*).
            5. **Lemmatization**: NLTK `WordNetLemmatizer` reduces words to root dictionary canonical form.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_matching:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<h4>2. Match Engine Metrics ({diag['engine']})</h4>", unsafe_allow_html=True)

            if "Groq" in diag["engine"]:
                st.markdown(f"**Groq LLM Reasoning:**")
                st.info(diag.get("reasoning", "N/A"))
                st.metric("Groq API Tokens Used", f"{diag.get('tokens_used', 0)}")
                st.markdown("---")

            if diag["scores"]:
                df_scores = pd.DataFrame(diag["scores"])
                df_scores["FAQ Short Title"] = df_scores["question"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)

                st.markdown("**Local Top FAQ Similarity Scores (TF-IDF vs Fuzzy Ratio):**")

                # Horizontal or bar chart
                df_plot = df_scores.sort_values(by="score", ascending=True)

                st.bar_chart(
                    data=df_plot,
                    x="FAQ Short Title",
                    y=["score", "tfidf_score", "fuzzy_score"],
                    use_container_width=True
                )

                with st.expander("Show detailed score calculation table"):
                    st.dataframe(
                        df_scores[["question", "score", "tfidf_score", "fuzzy_score", "confidence", "category"]].rename(
                            columns={
                                "question": "FAQ Question",
                                "score": "Hybrid Score",
                                "tfidf_score": "TF-IDF Score",
                                "fuzzy_score": "Fuzzy Ratio",
                                "confidence": "Confidence Tier",
                                "category": "Category"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    # Search Session Analytics Summary
    st.markdown("<h3 class='section-title'>📈 Search Session Analytics & Query Logs</h3>", unsafe_allow_html=True)
    if st.session_state.query_log:
        df_logs = pd.DataFrame(st.session_state.query_log)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Queries", len(df_logs))
        avg_score = df_logs["score"].mean()
        m2.metric("Avg Match Score", f"{avg_score:.3f}")
        high_conf = len(df_logs[df_logs["confidence"] == "High"])
        m3.metric("High Confidence Matches", f"{high_conf}")
        low_conf = len(df_logs[df_logs["confidence"].str.contains("Low")])
        m4.metric("Low / Unmatched Queries", f"{low_conf}")

        st.markdown("##### Query Execution Log")
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No queries executed in current session yet.")

# Tab 3: FAQ Database Manager
with tab3:
    st.markdown("<h3 class='section-title'>⚙️ FAQ Knowledge Base Administration</h3>", unsafe_allow_html=True)

    col_faqs_list, col_faqs_actions = st.columns([2.2, 1.2])

    with col_faqs_list:
        st.markdown("#### Current Knowledge Base Entries")

        col_srch, col_cat = st.columns([2, 1])
        search_query = col_srch.text_input("🔍 Search Database", "")
        all_cats = ["All"] + sorted(list(set([f.get("category", "General") for f in faqs])))
        filter_cat = col_cat.selectbox("Filter Category", all_cats)

        filtered_faqs = faqs
        if filter_cat != "All":
            filtered_faqs = [f for f in filtered_faqs if f.get("category", "General") == filter_cat]

        if search_query:
            sq = search_query.lower()
            filtered_faqs = [
                f for f in filtered_faqs
                if sq in f["question"].lower() or sq in f["answer"].lower() or sq in f.get("category", "").lower()
            ]

        if not filtered_faqs:
            st.warning("No FAQs found matching criteria.")
        else:
            for idx, faq in enumerate(filtered_faqs):
                cat = faq.get("category", "General")
                orig_idx = faqs.index(faq)

                with st.expander(f"[{cat}] {faq['question']}"):
                    if st.session_state.editing_idx == orig_idx:
                        # Edit mode
                        with st.form(key=f"edit_form_{orig_idx}"):
                            edit_q = st.text_input("Edit Question", value=faq['question'])
                            edit_a = st.text_area("Edit Answer", value=faq['answer'])
                            edit_cat = st.selectbox("Edit Category", ["General", "Account", "Billing", "Features", "Tools", "Security", "Support"], index=["General", "Account", "Billing", "Features", "Tools", "Security", "Support"].index(cat) if cat in ["General", "Account", "Billing", "Features", "Tools", "Security", "Support"] else 0)

                            c_save, c_cancel = st.columns([1, 1])
                            if c_save.form_submit_button("💾 Save Changes"):
                                faqs[orig_idx] = {"question": edit_q.strip(), "answer": edit_a.strip(), "category": edit_cat}
                                save_faqs(faqs)
                                st.session_state.editing_idx = None
                                st.success("FAQ updated successfully!")
                                st.rerun()

                            if c_cancel.form_submit_button("Cancel"):
                                st.session_state.editing_idx = None
                                st.rerun()
                    else:
                        st.markdown(f"**Question:** {faq['question']}")
                        st.markdown(f"**Answer:** {faq['answer']}")
                        st.markdown(f"**Category:** `{cat}`")

                        btn_col1, btn_col2 = st.columns([1, 1])
                        if btn_col1.button("✏️ Edit", key=f"btn_edit_{orig_idx}"):
                            st.session_state.editing_idx = orig_idx
                            st.rerun()

                        if btn_col2.button("🗑️ Delete", key=f"btn_del_{orig_idx}"):
                            faqs.pop(orig_idx)
                            save_faqs(faqs)
                            st.success("FAQ entry removed!")
                            st.rerun()

    with col_faqs_actions:
        st.markdown("#### ➕ Add New FAQ")
        with st.form("add_faq_form", clear_on_submit=True):
            new_q = st.text_input("Question", placeholder="e.g., What security protocols are used?")
            new_a = st.text_area("Answer", placeholder="e.g., We use TLS 1.3 and AES-256 GCM encryption...")
            new_c = st.selectbox("Category", ["General", "Account", "Billing", "Features", "Tools", "Security", "Support"])
            if st.form_submit_button("➕ Submit FAQ Entry"):
                if not new_q.strip() or not new_a.strip():
                    st.error("Question and Answer cannot be empty.")
                else:
                    faqs.append({"question": new_q.strip(), "answer": new_a.strip(), "category": new_c})
                    save_faqs(faqs)
                    st.success("FAQ added successfully!")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 🔄 Import / Export Knowledge Base")

        # Export JSON / CSV
        export_json = json.dumps(faqs, indent=2)
        st.download_button(
            label="📦 Export FAQs (JSON)",
            data=export_json,
            file_name="faqs_export.json",
            mime="application/json",
            use_container_width=True
        )

        df_export = pd.DataFrame(faqs)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Export FAQs (CSV)",
            data=csv_buffer.getvalue(),
            file_name="faqs_export.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Import File
        uploaded_file = st.file_uploader("Upload FAQ File (JSON or CSV)", type=["json", "csv"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".json"):
                    imported_data = json.load(uploaded_file)
                else:
                    df_imp = pd.read_csv(uploaded_file)
                    imported_data = df_imp.to_dict(orient="records")

                if isinstance(imported_data, list) and len(imported_data) > 0 and "question" in imported_data[0]:
                    if st.button("Overwrite & Import Knowledge Base"):
                        save_faqs(imported_data)
                        st.success(f"Successfully imported {len(imported_data)} FAQs!")
                        st.rerun()
                else:
                    st.error("Invalid file structure. Must contain 'question' and 'answer' fields.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
