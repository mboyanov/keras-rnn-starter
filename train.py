from preprocessing.preprocesser import Preprocesser
from models.model import getModel
import numpy as np
from keras.optimizers import Adam

train_file = ""
test_file = ""

def read(f):
    return ["I am banana", "you are banana"], np.array([[1,0], [0,1]])

xs , ys = read(train_file)
xs_test, ys_test = read(test_file)
preprocesser = Preprocesser(10, 10)

xs_seq = preprocesser.preprocess(xs)
xs_test_seq = preprocesser.preprocess(xs_test, train=False)

model = getModel(10, 10)
model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

for i in range(1000):
    model.fit(xs_seq, ys)
    results = model.evaluate(xs_test_seq, ys_test)