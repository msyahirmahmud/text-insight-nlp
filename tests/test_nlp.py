import unittest
from nlp_engine import TextInsightNLP

class TestTextInsightNLP(unittest.TestCase):
    def setUp(self):
        self.nlp = TextInsightNLP()

    def test_estimate_reading_time_calculates_minutes(self):
        text = " ".join(["word"] * 400)
        res = self.nlp.estimate_reading_time(text, wpm=200)
        self.assertEqual(res["word_count"], 400)
        self.assertEqual(res["minutes"], 2.0)
        self.assertEqual(res["formatted"], "2 min read")

    def test_positive_sentiment_detection(self):
        text = "This new software release is fast, clean, and scalable!"
        res = self.nlp.analyze_sentiment(text)
        self.assertEqual(res["label"], "POSITIVE")
        self.assertGreater(res["positive_count"], 0)

    def test_negative_sentiment_detection(self):
        text = "The application crashed with a terrible slow bug error."
        res = self.nlp.analyze_sentiment(text)
        self.assertEqual(res["label"], "NEGATIVE")

    def test_readability_metrics_calculation(self):
        text = "Python is a high-level programming language. It is easy to learn and write."
        res = self.nlp.calculate_readability(text)
        self.assertGreater(res["words"], 0)
        self.assertEqual(res["sentences"], 2)
        self.assertGreater(res["flesch_score"], 0)

if __name__ == '__main__':
    unittest.main()
