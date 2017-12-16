from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model


def getModel(vocab_size, seq_length) -> Model:
    main_input = Input(shape=(seq_length,), dtype='int32', name='main_input')
    x = Embedding(output_dim=128, input_dim=vocab_size, input_length=seq_length, mask_zero=True)(main_input)
    lstm_out = LSTM(32)(x)
    out = Dense(2, activation="sigmoid")(lstm_out)
    model = Model(inputs=[main_input], outputs=[out])
    return model
