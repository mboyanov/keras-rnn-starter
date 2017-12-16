from preprocessing.preprocesser import Preprocesser
from models.model import getModel
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
train_file = ""
test_file = ""

def read(f):
    return ["I am banana", "you are banana"], np.array([[1,0], [0,1]])

xs , ys = read(train_file)
xs_test, ys_test = read(test_file)
preprocesser = Preprocesser(10, 10)

xs_seq = preprocesser.preprocess(xs)
xs_test_seq = preprocesser.preprocess(xs_test, train=False)

callbacks = [TensorBoard(log_dir="/tmp/logs/", embeddings_freq=1, embeddings_layer_names=["embedding"], embeddings_metadata={
    "embedding": preprocesser.vocab_to_file("/tmp/vocab")
})]

model = getModel(10, 10)
model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(xs_seq, ys, epochs=100, callbacks=callbacks, validation_data=(xs_test_seq, ys_test))