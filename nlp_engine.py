"""
Text Insight NLP Analytics Engine (Python)
"""

import re
from collections import Counter

class TextInsightNLP:
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "is", "in", "it", "of", "to", "for", "with", "on", "at", "by"}

    def analyze_sentiment(self, text: str) -> dict:
        if not text or not text.strip():
            return {"score": 0.0, "label": "NEUTRAL"}

        positive_words = {"good", "great", "excellent", "awesome", "fast", "clean", "easy", "love", "scalable"}
        negative_words = {"bad", "slow", "bug", "error", "fail", "terrible", "hard", "crash"}

        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)

        net_score = pos_count - neg_count
        label = "NEUTRAL"
        if net_score > 0:
            label = "POSITIVE"
        elif net_score < 0:
            label = "NEGATIVE"

        return {
            "positive_count": pos_count,
            "negative_count": neg_count,
            "score": net_score,
            "label": label
        }

    def get_top_ngrams(self, text: str, n: int = 2, top_k: int = 5) -> list:
        if not text:
            return []
        words = [w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in self.stop_words]
        if len(words) < n:
            return []
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        return counts.most_common(top_k)

    def estimate_reading_time(self, text: str, wpm: int = 200) -> dict:
        if not text:
            return {"word_count": 0, "minutes": 0.0, "formatted": "0 min read"}
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        minutes = round(word_count / wpm, 2)
        display_minutes = max(1, round(minutes))
        return {
            "word_count": word_count,
            "minutes": minutes,
            "formatted": f"{display_minutes} min read"
        }

    def calculate_readability(self, text: str) -> dict:
        if not text or not text.strip():
            return {"words": 0, "sentences": 0, "score": 100.0}

        words = re.findall(r'\b\w+\b', text)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]

        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        avg_sentence_length = word_count / sentence_count
        score = max(0.0, min(100.0, round(206.835 - (1.015 * avg_sentence_length), 2)))

        return {
            "words": word_count,
            "sentences": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "flesch_score": score
        }
