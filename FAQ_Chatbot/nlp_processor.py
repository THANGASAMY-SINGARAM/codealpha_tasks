import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
from groq import Groq
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def download_nltk_resources():
    """Download required NLTK resources safely if not already present."""
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            if res == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

# Run download safely
download_nltk_resources()


def preprocess_text(text):
    """
    Lowercase, tokenize, remove punctuation and stopwords, and lemmatize words.
    Returns (preprocessed_string, token_list).
    """
    if not text or not isinstance(text, str):
        return "", []

    # Lowercase
    text = text.lower().strip()
    if not text:
        return "", []

    # Tokenize safely
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Remove punctuation and non-alphanumeric tokens
    table = str.maketrans('', '', string.punctuation)
    tokens = [t.translate(table) for t in tokens if t.translate(table)]

    # Stopwords removal
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {"is", "the", "a", "an", "and", "or", "to", "in", "on", "of", "for", "with", "it"}

    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        pass

    return " ".join(tokens), tokens


def calculate_fuzzy_score(seq1, seq2):
    """
    Calculates SequenceMatcher ratio between two preprocessed strings or raw queries.
    """
    if not seq1 or not seq2:
        return 0.0
    return float(SequenceMatcher(None, seq1.lower(), seq2.lower()).ratio())


def get_confidence_tier(score):
    """
    Returns a human-readable confidence label and color code based on match score.
    """
    if score >= 0.60:
        return {"level": "High", "color": "#00e676"}
    elif score >= 0.35:
        return {"level": "Medium", "color": "#ff9100"}
    else:
        return {"level": "Low / Unmatched", "color": "#ff5252"}


def match_faq_local(user_query, faqs, threshold=0.3, tfidf_weight=0.65, fuzzy_weight=0.35):
    """
    Matches user query against FAQs using a Hybrid approach:
    TF-IDF Cosine Similarity + Fuzzy Levenshtein Ratio.
    """
    if not faqs or not user_query or not user_query.strip():
        return None, 0.0, [], []

    questions = [faq['question'] for faq in faqs]
    preprocessed_questions = [preprocess_text(q)[0] for q in questions]
    preprocessed_query, query_tokens = preprocess_text(user_query)

    # Fallback to raw lowercase query if tokenization cleared everything
    clean_query = preprocessed_query if preprocessed_query else user_query.lower().strip()

    # Compute TF-IDF Cosine Similarity
    corpus = [clean_query] + [pq if pq else q.lower() for pq, q in zip(preprocessed_questions, questions)]
    
    tfidf_scores = np.zeros(len(faqs))
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[0]
        faq_vectors = tfidf_matrix[1:]
        tfidf_scores = cosine_similarity(query_vector, faq_vectors).flatten()
    except ValueError:
        # Happens when vocabulary is empty or only unigram stop words
        pass

    # Compute Fuzzy Scores
    fuzzy_scores = np.array([
        max(
            calculate_fuzzy_score(user_query, q),
            calculate_fuzzy_score(clean_query, pq)
        )
        for q, pq in zip(questions, preprocessed_questions)
    ])

    # Compute Hybrid Scores
    hybrid_scores = (tfidf_weight * tfidf_scores) + (fuzzy_weight * fuzzy_scores)

    # Format result list
    all_scores = []
    for idx, (faq, tf_score, fz_score, hy_score) in enumerate(zip(faqs, tfidf_scores, fuzzy_scores, hybrid_scores)):
        conf = get_confidence_tier(float(hy_score))
        all_scores.append({
            "index": idx,
            "question": faq['question'],
            "score": float(hy_score),
            "tfidf_score": float(tf_score),
            "fuzzy_score": float(fz_score),
            "category": faq.get('category', 'General'),
            "confidence": conf['level'],
            "confidence_color": conf['color']
        })

    # Sort descending
    all_scores = sorted(all_scores, key=lambda x: x['score'], reverse=True)

    best_match_idx = all_scores[0]['index']
    best_score = all_scores[0]['score']

    if best_score >= threshold:
        return faqs[best_match_idx], best_score, all_scores, query_tokens
    else:
        return None, best_score, all_scores, query_tokens


def match_faq_groq(user_query, faqs, conversation_history=None, api_key=None):
    """
    Submits user query, FAQ database, and optional past context to Groq LLM (llama-3.1-8b-instant).
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "answer": "Groq API key not found. Please set the GROQ_API_KEY in the sidebar or in your `.env` file.",
            "matched_question": "N/A",
            "reasoning": "Missing API Key",
            "tokens_used": 0
        }

    try:
        client = Groq(api_key=api_key)

        faq_context = ""
        for idx, faq in enumerate(faqs):
            faq_context += f"[{idx}] Category: {faq.get('category', 'General')} | Q: {faq['question']} | A: {faq['answer']}\n"

        history_context = ""
        if conversation_history:
            history_snippets = []
            for msg in conversation_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_snippets.append(f"{role}: {msg['content']}")
            history_context = "\nRecent Conversation History:\n" + "\n".join(history_snippets) + "\n"

        system_prompt = f"""You are an intelligent AI customer support assistant for Aether Cloud Platform.
You have access to a reference FAQ Knowledge Base and past chat context.

Goal:
1. Determine if the user's question can be answered using the reference FAQs or previous context.
2. If yes: select the best matching FAQ, and compose a helpful, polite, and precise response grounded ONLY in that FAQ's answer.
3. If the user asks a follow-up, use past conversation history to maintain context.
4. If no FAQ matches: state politely that you couldn't find a direct answer, suggest the closest relevant FAQ if available, and provide general guidance.

FAQ Knowledge Base:
{faq_context}
{history_context}
Output MUST be a JSON object with keys:
- "matched_index": integer index of matched FAQ (-1 if no match)
- "matched_question": exact question string matched ("None" if no match)
- "answer": your friendly final response
- "reasoning": concise explanation of why this match was chosen
"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.2
        )

        response_text = chat_completion.choices[0].message.content
        res_json = json.loads(response_text)

        usage = chat_completion.usage
        tokens_used = usage.total_tokens if usage else 0
        res_json["tokens_used"] = tokens_used

        return res_json

    except Exception as e:
        return {
            "answer": f"Error communicating with Groq API: {str(e)}",
            "matched_question": "N/A",
            "reasoning": f"Exception: {type(e).__name__}",
            "tokens_used": 0
        }
