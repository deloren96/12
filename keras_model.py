#from pr import roll
import numpy as np
import keras
import tensorflow as tf
import tensorflow.keras as k


inputs = k.Input(shape=(2,))
x = k.layers.Dense(2, activation=k.activations.sigmoid, weights=[np.array([[3.459213722, -6.31474346], [0.2616570362, 4.853637179]]), np.array([0., 0.])])(inputs)

outputs = k.layers.Dense(1, activation=k.activations.sigmoid, weights=[np.array([[3.135244311], [-6.517022925]]), np.array([0.])])(x)

model = k.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="Adam")
#model.summary()
for layer in model.layers: print(layer, layer.get_weights())

y = model.get_weights()
print(y)

#model.fit()

a = np.random.uniform(0,1, (1, 2))
print(a, model(a))
#sk.utils.plot_model(model, 'my_first_model.png')