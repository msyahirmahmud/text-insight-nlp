from nlp import TextInsightNLP

def main():
    nlp = TextInsightNLP()
    sample_text = "Building modern web applications with Node.js and Python is great! Fast execution and clean code."
    
    print("🧠 TextInsight NLP Analysis Results:")
    print("-----------------------------------")
    print("Sentiment:", nlp.analyze_sentiment(sample_text))
    print("Keywords:", nlp.extract_keywords(sample_text))
    print("Readability:", nlp.compute_readability_score(sample_text))

if __name__ == "__main__":
    main()
