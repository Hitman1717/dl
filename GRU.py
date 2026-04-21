import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

vocab_size=10000
max_length=200

(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=vocab_size)

x_train=keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_length)
x_test=keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_length)

model=keras.Sequential([
    layers.Embedding(vocab_size,128,input_length=max_length),
    layers.GRU(64),
    layers.Dense(64,activation="relu"),
    layers.Dense(1,activation="sigmoid")
    ])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

history_gru=model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2
    )

loss,acc=model.evaluate(x_test,y_test)
print(acc)
