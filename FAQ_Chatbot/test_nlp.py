import pytest
import json
import os
from nlp_processor import (
    preprocess_text,
    calculate_fuzzy_score,
    get_confidence_tier,
    match_faq_local,
    match_faq_groq
)

@pytest.fixture
def sample_faqs():
    return [
        {
            "question": "What is the Aether Cloud Platform?",
            "answer": "Aether Cloud Platform is a next-generation decentralized cloud infrastructure.",
            "category": "General"
        },
        {
            "question": "How do I sign up for an Aether account?",
            "answer": "You can sign up by visiting our portal at console.aether.io.",
            "category": "Account"
        },
        {
            "question": "What is your refund policy?",
            "answer": "We offer a 30-day money-back guarantee.",
            "category": "Billing"
        }
    ]

def test_preprocess_text_normal():
    clean, tokens = preprocess_text("How do I sign up for an account?")
    assert "sign" in tokens
    assert "account" in tokens
    # Stop words like 'how', 'do', 'i', 'for', 'an' should be removed
    assert "how" not in tokens
    assert "an" not in tokens

def test_preprocess_text_empty_and_special():
    clean, tokens = preprocess_text("")
    assert clean == ""
    assert tokens == []

    clean, tokens = preprocess_text("!!! ??? ###")
    assert clean == ""
    assert tokens == []

def test_calculate_fuzzy_score():
    score_exact = calculate_fuzzy_score("refund policy", "refund policy")
    assert score_exact == 1.0

    score_partial = calculate_fuzzy_score("refnd policy", "refund policy")
    assert score_partial > 0.80

    score_different = calculate_fuzzy_score("xyzabc", "refund policy")
    assert score_different < 0.30

def test_get_confidence_tier():
    assert get_confidence_tier(0.85)["level"] == "High"
    assert get_confidence_tier(0.45)["level"] == "Medium"
    assert get_confidence_tier(0.15)["level"] == "Low / Unmatched"

def test_match_faq_local_exact(sample_faqs):
    match, score, scores, tokens = match_faq_local("What is the Aether Cloud Platform?", sample_faqs)
    assert match is not None
    assert match["question"] == "What is the Aether Cloud Platform?"
    assert score > 0.80

def test_match_faq_local_typos(sample_faqs):
    match, score, scores, tokens = match_faq_local("refnd policyyy", sample_faqs, threshold=0.30)
    assert match is not None
    assert match["question"] == "What is your refund policy?"

def test_match_faq_local_out_of_scope(sample_faqs):
    match, score, scores, tokens = match_faq_local("What is the recipe for chocolate cake?", sample_faqs, threshold=0.50)
    assert match is None
    assert score < 0.50

def test_match_faq_groq_missing_key(sample_faqs):
    res = match_faq_groq("test query", sample_faqs, api_key="")
    assert "answer" in res
    assert "matched_question" in res
    assert res["matched_question"] in ["N/A", "None"]
