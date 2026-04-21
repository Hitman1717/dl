from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.datasets import cifar10
#optional to show image
import matplotlib.pyplot as plt
import numpy as np

(x_train , y_train),(x_test , y_test) = cifar10.load_data()

# (x_train , y_train),(x_test , y_test) = cifar10.load_data(  path = " D/sample.npz ")
# this to give custom path if data given by pen drive 

x_train = x_train/255.0
x_test = x_test/255.0

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

model = Sequential([
    Conv2D(32, (3,3) , activation='relu' , input_shape = (32,32,3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3) , activation="relu" ),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    
    Flatten(),
    Dense(64,activation="relu"),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)



# Show image
plt.imshow(x_test[0])
plt.axis('off')

# Prediction
pred = model.predict(x_test)
pred_class = np.argmax(pred[0])
true_class = int(y_test[0])

print("Predicted:", class_names[pred_class])
print("Actual   :", class_names[true_class])