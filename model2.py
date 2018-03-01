#!/usr/bin/env python

"model2: convolution only across bases"

import os
import keras
import random
import numpy as np

def get_model():
    input1 = keras.layers.Input(shape=(256, 63, 10))
    input2 = keras.layers.Input(shape=(256,  1, 10))
    input3 = keras.layers.Input(shape=(2,))

    x = keras.layers.Convolution2D(16, (1, 3), activation="relu", padding="same")(input1)
    x = keras.layers.MaxPooling2D((1, 3), strides=(1, 3))(x)
    x = keras.layers.Convolution2D(32, (1, 5), activation="relu", padding="same")(x)
    x = keras.layers.Convolution2D(32, (1, 5), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((1, 3), strides=(1, 3))(x)
    conv = keras.layers.Convolution2D(32, (1, 7), activation="relu", padding="valid")(x)

    fc = keras.layers.Convolution2D(32, (1, 63), activation="relu", padding="valid")(input1)

    x = keras.layers.Concatenate(3)([conv, fc, input2])
    x = keras.layers.Convolution2D(64, (1, 1), activation="relu", padding="same")(x)
    x = keras.layers.AveragePooling2D((256, 1), strides=(256, 1))(x)
    res = keras.layers.Flatten()(x)

    x = keras.layers.Concatenate(1)([res, input3])
    x = keras.layers.Dense(32, activation="sigmoid")(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.models.Model(inputs=[input1, input2, input3], outputs=[x])
    model.compile(keras.optimizers.SGD(lr=1e-3, momentum=.95), "binary_crossentropy", metrics=["accuracy"])
    return model

def collect_data():
    files = os.listdir('.')
    images = [x[:-3] for x in files if x.endswith('.image')]
    samples = [x for x in images if x+".txt" in files]

    data = []
    for sam in samples:
        with open(sam+".image") as image, open(sam+".txt") as txt:
            for line in txt:
                line = line.split()
                y, freq, depth = int(line[3]), float(line[4]), int(line[5])
                X = np.zeros(dtype=float, shape=(256,64,10))
                reads = [np.fromfile(image, dtype=float, count=64*10)]
                if depth > 256:
                    reads = [reads[x] for x in sorted(random.sample(range(depth), count=256))]
                for i in range(len(reads)):
                    X[i, :, :] = read
                data.append((X[:, 0:63, :], X[:, [63], :], np.array([freq, min(depth, 2048)/depth]), y))

    with open("model2.data", "w") as out:
        np.array([len(data)]).tofile(out)
        np.array([d[3] for d in data]).tofile(out)
        for X, _, _, _ in data:
            X.tofile(out)
        for _, X, _, _ in data:
            X.tofile(out)
        for _, _, X, _ in data:
            X.tofile(out)

def train():
    with open("model1.data") as f:
        n = np.fromfile(f, dtype=int, count=1)[0]
        y = np.fromfile(f, dtype=int, count=n)
        X1 = np.fromfile(f, dtype=float, count=n*256*63*10).reshape(n, 256, 63, 10)
        X2 = np.fromfile(f, dtype=float, count=n*256*10).reshape(n, 256, 1, 10)
        X3 = np.fromfile(f, dtype=float, count=n*2).reshape(n, 2)

    model = get_model()
    callbacks = [keras.callbacks.ModelCheckpoint("model2_weight.{epoch:02d}-{val_acc:.4f}.h5", monitor="val_acc")]
    
    model.fit(X, y, batch_size=64, epochs=20, validation_split=.02, callbacks=callbacks)

if __name__ == '__main__':
    import fire
    fire.Fire({ 'train': train, 'collect_data': collect_data })