import random

import numpy as np

from pr import *
from time import time
import keras
import tensorflow as tf
import tensorflow.keras as k
try:
    model = k.models.load_model('./')
except:
    inputs = k.Input(shape=(2,))
    x = k.layers.Dense(5, activation=k.activations.sigmoid, )(inputs)#weights=[np.array([[3.459213722, -6.31474346], [0.2616570362, 4.853637179]]), np.array([0., 0.])]
    x = k.layers.Dense(3, activation=k.activations.sigmoid, )(x) #weights=[np.array([[3.135244311], [-6.517022925]]), np.array([0.])]
    outputs = k.layers.Dense(1, activation=k.activations.sigmoid, )(x) #weights=[np.array([[3.135244311], [-6.517022925]]), np.array([0.])]

    model = k.Model(inputs=inputs, outputs=outputs)

model.compile()
#model.summary()
#for layer in model.layers: print(layer, layer.get_weights())

popsize = 150
def fillpop():
    while len(pop)!=popsize:
        pop.append([])
        for i in model.layers:
            wi = i.get_weights()
            #print(i, wi)
            if wi:

                pop[-1].append(k.initializers.RandomUniform(minval=-1., maxval=1.)(shape=(wi[0].shape)))
                pop[-1].append(np.zeros(wi[1].size))
    #print(1,len(pop), "\nm",)
try:
    pop = np.load('./model.npy').tolist()
except:
    pop = []
    fillpop()

old = []
def cut_w():
    global old
    old = []
    for j in range(len(pop)):
        arrays = []
        model.set_weights(pop[j])
        y = model.get_weights()
        for i in range(0, len(y), 2):
            for k in range(0, len(y[i])):
                arrays+=y[i][k].tolist()
        old.append([arrays, play()])
        #print(old[-1])
    old.sort( key=lambda x: (x[1][0], x[1][1]), reverse=True)
    print(old[0][1])

def mutation(chl):
    for i in range(0, len(chl)):
        if random.random() < 1/len(chl):
            chl[i] = 1 - chl[i]
        return chl

def crossover():
    global pop
    pop = []

    for k in range(0, int(len(old)*0.30)):
        pop.append(old[k][0])
    # print(pop)
    for i in range(0, len(old), 2):
        r = random.randint(0,len(old[i])-2)
        pop.append(mutation(random.choice([old[i][0][r:]+old[i][0][:r], old[i+1][0][:r]+old[i+1][0][r:]])))

    for k in range(0, int(len(pop))):
        popk = pop[k][:]
        pop[k] = []
        for i in model.layers:
            wi = i.get_weights()
            # print(i, wi)
            if wi:
                pop[k].append(np.reshape(np.array(popk[:wi[0].size]), wi[0].shape))
                del popk[:wi[0].size]
                pop[k].append(np.zeros(wi[1].size))
                # print(pop[-1])
    fillpop()
    # print(len(pop))
#model.fit()
def play():
    #print('total:', sc)
    l = np.array([3,6])
    sc = 0
    minimum = 0
    #play()
    attempt = 0
    while sc < 100000 and attempt < 1000:
        attempt+=1
        #print(sc)
        #print(attempt)
        t, l,b = roll(attempt, l)
        #a = np.random.uniform(0,1, (1, 2))
        res =  model(t).numpy()[0]
        for (i, v) in enumerate(res):
            #print('res0', res)
            res[i] = int(((10000-20)*res[i]) + 20 + 0.5)
        sc = sc+res[0] if b ==0 else sc-res[0]
        #print('res', res)
        #print(np.int64(res))
        if sc<minimum:
            minimum = sc
    # print('Попыток:', attempt, 'Дней:', round(attempt/1440, 2), minimum)
    return sc, minimum
while True:
    tim = time()

    cut_w()
    crossover()
    print(time()-tim)
    np.save('./model.npy', np.array(pop, dtype=object))
# if sc > 1000:
#     model.save('./')
    #sk.utils.plot_model(model, 'my_first_model.png')


