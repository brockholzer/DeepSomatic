#!/usr/bin/env python

"model1: traditional CNN"

import os
import keras
import random
import numpy as np

def get_model():
    model = keras.models.Sequential([
        keras.Input(shape=(256, 64, 10)),

        keras.layers.Convolution2D(16, (5, 5), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Convolution2D(32, (5, 5), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Convolution2D(64, (1, 5), activation="relu", padding="same"),
        keras.layers.Convolution2D(64, (5, 1), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((4, 1), strides=(4, 1)),

        keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid")
    ])

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
                y, depth = int(line[3]), int(line[5])
                X = np.zeros(dtype=float, shape=(256,64,10))
                reads = [np.fromfile(image, dtype=float, count=64*10)]
                if depth > 256:
                    reads = [reads[x] for x in sorted(random.sample(range(depth), count=256))]
                for i in range(len(reads)):
                    X[i, :, :] = read
                data.append((X, y))

    with open("model1.data", "w") as out:
        np.array([len(data)]).tofile(out)
        np.array([y for _, y in data]).tofile(out)
        for X, _ in data:
            X.tofile(out)

def train():
    with open("model1.data") as f:
        n = np.fromfile(f, dtype=int, count=1)[0]
        y = np.fromfile(f, dtype=int, count=n)
        X = np.fromfile(f, dtype=float, count=n*256*64*10).reshape(n, 256, 64, 10)

    model = get_model()
    callbacks = [keras.callbacks.ModelCheckpoint("model1_weight.{epoch:02d}-{val_acc:.4f}.h5", monitor="val_acc")]
    
    model.fit(X, y, batch_size=64, epochs=20, validation_split=.02, callbacks=callbacks)

if __name__ == '__main__':
    import fire
    fire.Fire({ 'train': train, 'collect_data': collect_data })