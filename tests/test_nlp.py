import unittest
from nlp import TextInsightNLP

class TestTextInsightNLP(unittest.TestCase):
    def setUp(self):
        self.nlp = TextInsightNLP()

    def test_positive_sentiment(self):
        text = "This product is amazing, fast, and great!"
        res = self.nlp.analyze_sentiment(text)
        self.assertEqual(res["sentiment"], "positive")
        self.assertGreater(res["score"], 0)

    def test_negative_sentiment(self):
        text = "The application is terrible, broken, and slow."
        res = self.nlp.analyze_sentiment(text)
        self.assertEqual(res["sentiment"], "negative")
        self.assertLess(res["score"], 0)

    def test_extract_keywords(self):
        text = "Python is a powerful programming language. Python makes software engineering easy."
        keywords = self.nlp.extract_keywords(text, top_n=2)
        self.assertIn("python", keywords)

    def test_readability_metrics(self):
        text = "First sentence here. Second sentence follows!"
        metrics = self.nlp.compute_readability_score(text)
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertEqual(metrics["word_count"], 6)

if __name__ == '__main__':
    unittest.main()
