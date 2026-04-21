# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:34:09 2026

@author: Rohit
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

#y=2x-1

model=keras.Sequential([
    layers.Dense(units=1,input_shape=[1])
    ])

model.compile(optimizer="sgd",
              loss="mean_squared_error"
              )
xs=np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],dtype=float)
ys=np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0],dtype=float)


model.fit(xs,ys,epochs=20)

input_data=np.array([10.0])

print(model.predict(input_data))