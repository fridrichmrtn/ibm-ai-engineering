'''
Making sure emotion detection works.
'''

import unittest
from emotion_detection import emotion_detector

class EmotionTest(unittest.TestCase):
    '''
        Compendium of emotional tests.
    '''
    def test_dominant_emotions(self):
        '''
            Check for dominant emotions.
        '''
        self.assertEqual(
            emotion_detector("I hate working long hours.")["dominant_emotion"], "anger")
        self.assertEqual(
            emotion_detector("I am glad this happened.")["dominant_emotion"], "joy")
        self.assertEqual(
            emotion_detector("I am really mad about this.")["dominant_emotion"], "anger")
        self.assertEqual(
            emotion_detector("I feel disgusted just hearing about this.")["dominant_emotion"], "disgust")                                    
        self.assertEqual(
            emotion_detector("I am so sad about this.")["dominant_emotion"], "sadness")
        self.assertEqual(
            emotion_detector("I am really afraid that this will happen.")["dominant_emotion"], "fear")                                    

unittest.main()
