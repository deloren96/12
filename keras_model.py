from pr import *
import keras
import tensorflow as tf
import tensorflow.keras as k
try:
    model = k.models.load_model('./')
except:
    inputs = k.Input(shape=(37,))
    x = k.layers.Dense(37, activation=k.activations.sigmoid, kernel_initializer=k.initializers.RandomUniform(minval=-1., maxval=1.))(inputs)#weights=[np.array([[3.459213722, -6.31474346], [0.2616570362, 4.853637179]]), np.array([0., 0.])]

    outputs = k.layers.Dense(37, activation=k.activations.sigmoid, kernel_initializer=k.initializers.RandomUniform(minval=-1., maxval=1.))(x) #weights=[np.array([[3.135244311], [-6.517022925]]), np.array([0.])]

    model = k.Model(inputs=inputs, outputs=outputs)

    model.compile()
#model.summary()
#for layer in model.layers: print(layer, layer.get_weights())

y = model.get_weights()
#print(y)
l = np.ones(37)
sc = 0
minimum = 0
#model.fit()
def play():
    global sc, l, minimum
    for i in range(1, 1*1+1):
        t, l,b = roll(i, l)
        #a = np.random.uniform(0,1, (1, 2))
        res =  model(t).numpy()[0]
        #print(res)
        for (i, v) in enumerate(res):
            res[i] = int(res[i]*10 + 0.5)
        sc+=score(b, res)
        #print(np.int64(res))
    if sc<minimum:
        minimum = sc
    #print('total:', sc)
for _ in range(1, 100+1):
    l = np.ones(37)
    sc = 0
    minimum = 0
    play()
    attempt = 0
    while sc < 1000:
        attempt+=1
        play()
    print('Попыток:', attempt, 'Дней:', round(attempt/1440, 2), minimum)
#if sc > 100000:
    #model.save('/mnt/c/_DW/story of/GitHub/12')
        #sk.utils.plot_model(model, 'my_first_model.png')