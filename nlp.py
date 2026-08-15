"""
TextInsight NLP Engine - Keyword Extraction, Readability Scoring, and Sentiment Analysis
"""

import re
from collections import Counter

class TextInsightNLP:
    POSITIVE_WORDS = {'great', 'excellent', 'amazing', 'good', 'awesome', 'outstanding', 'love', 'fast', 'smooth'}
    NEGATIVE_WORDS = {'bad', 'poor', 'terrible', 'slow', 'broken', 'bug', 'error', 'worst', 'hate', 'fail'}

    def analyze_sentiment(self, text: str) -> dict:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {"sentiment": "neutral", "score": 0.0}

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)

        total = pos_count + neg_count
        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2: label = "positive"
            elif score < -0.2: label = "negative"
            else: label = "neutral"

        return {
            "sentiment": label,
            "score": round(score, 2),
            "positive_words": pos_count,
            "negative_words": neg_count
        }

    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        stopwords = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'it', 'this', 'that'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        counter = Counter(filtered)
        return [item[0] for item in counter.most_common(top_n)]

    def compute_readability_score(self, text: str) -> dict:
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        if not sentences or not words:
            return {"word_count": 0, "sentence_count": 0, "avg_words_per_sentence": 0}

        avg_wps = len(words) / len(sentences)
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": round(avg_wps, 1)
        }
