import unittest
from sentiment_analysis import sentiment_analyzer

class TestSentiment(unittest.TestCase):

    def test_score(self):
        self.assertTrue(sentiment_analyzer("I love python")["score"]>0)
        self.assertTrue(sentiment_analyzer("I hate python")["score"]<0)
        self.assertTrue(int(sentiment_analyzer("I am neutral to python")["score"])==0)

    def test_label(self):
        self.assertTrue(sentiment_analyzer("I love python")["label"]=="SENT_POSITIVE")
        self.assertTrue(sentiment_analyzer("I hate python")["label"]=="SENT_NEGATIVE")
        self.assertTrue(sentiment_analyzer("I am neutral to python")["label"]=="SENT_NEUTRAL")     

unittest.main() 

    