from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.preprocessing.text import text_to_word_sequence

import numpy as np

_UNK_= 1
_PAD_ = 0

class Preprocesser:

    def __init__(self, vocab_size, maxlen, w2id= None):
        self.vocab_size = vocab_size
        self.maxlen= maxlen
        if w2id is None:
            self.w2id = {}
        else:
            self.w2id = w2id
        self.id2word = {y: x for x,y in self.w2id.items()}

    def tokenize(self, text):
        return text_to_word_sequence(text)

    def fit_on_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            for w in self.tokenize(text):
                word_counts[w] += 1
        vocab = [x[0] for x in word_counts.most_common(self.vocab_size-2)] # Leave space for 0 and UNK
        self.w2id = {w: i+2 for i, w in enumerate(vocab)}
        self.id2word = {y: x for x, y in self.w2id.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = np.array([self.w2id.get(w, _UNK_) for w in self.tokenize(text)])
            sequences.append(seq)
        return np.array(sequences)

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            texts.append(" ".join([self.id2word[x] for x in seq if x > 0]))
        return texts


    def preprocess(self, raw_texts: list, train: bool=True):
        """
        converts raw_texts to sequences
        :param raw_texts:
        :param train: if set to true, will fit the index.
        :return:
        """
        if train:
            self.fit_on_texts(raw_texts)
        seq = self.texts_to_sequences(raw_texts)
        return pad_sequences(seq, maxlen=self.maxlen, value=_PAD_)
