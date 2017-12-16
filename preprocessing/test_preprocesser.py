from unittest import TestCase
from preprocessing.preprocesser import Preprocesser
import numpy as np
class TestPreprocesser(TestCase):


    def test_preprocess(self):
        preprocesser = Preprocesser(5, 4)
        texts = ["I am banana",
                 "I like apples",
                 "Apples are good for your health",
                 "What is the meaning of life?",
                 "How is your snack?"]
        res = preprocesser.preprocess(texts)
        self.assertEqual(res.shape, (5,4))
        self.assertEqual(np.max(res), 4)
