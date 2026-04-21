import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

model=keras.Sequential([
    layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
    ])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))

loss,acc=model.evaluate(x_test,y_test)
print("accuracy :",acc)

prediction=model.predict(x_test)

i=0

pred_label=np.argmax(prediction[i])
true_label=y_test[i]

plt.imshow(x_test[i].reshape(28,28),cmap='gray')

plt.title(f"predicted: {pred_label} | actual: {true_label} ")

plt.axis('off')
plt.show()