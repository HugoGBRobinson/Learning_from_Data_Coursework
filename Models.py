import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import matplotlib

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import math

wine_df = pd.read_csv('winequality-red.csv', sep=";")

target_data = wine_df['quality']
del wine_df['quality']
values = wine_df

train_features, test_features, train_targets, test_targets = train_test_split(values, target_data, test_size=0.2)


def quality_values(target_data, values):
    x = target_data
    y = values['alcohol']

    # Normalized data to allow for more accurate gradients
    y_max = y.max()
    normalized_y = y / y_max

    plt.scatter(x, normalized_y)

    z = np.polyfit(x, normalized_y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    print(z)

    plt.title("Alcohol: Gradient = " + str(round(z[0], 3)))
    plt.ylabel("Normalised Alcohol")
    plt.xlabel("Quality")
    plt.show()


# fixed acidity = small positive correlation = 0.017
# volatile acidity = negative correlation = -0.055
# citric acid = positive correlation = 0.055
# residual sugar = very small positive correlation, outliers = 0.002
# chlorides = very small negative correlation, outliers = -0.012
# free sulfur dioxide = very large negative correlation = -0.009
# total sulfur dioxide = very large negative correlation, notable outliers = -0.026
# density = small negative correlation = 0.0
# pH = very small negative correlation, some outliers = -0.003
# sulphates = positive correlation, many outliers = 0.026
# alcohol = strong positive correlation = 0.042

def CNN(train_features, test_features, train_targets, test_targets, values, target_data):
    classes_num = 10

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(np.prod(train_features.shape[1:]),)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_features, train_targets, batch_size=256, epochs=100,
                        verbose=1, validation_data=(test_features, test_targets))

    [test_loss, test_acc] = model.evaluate(test_features, test_targets)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

    # Plot the Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    plt.show()

    # Plot the Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    plt.show()


def MLPModel(train_features, test_features, train_targets, test_targets):
    classifier = MLPClassifier(hidden_layer_sizes=7, max_iter=100, activation='relu', solver='sgd', verbose=10,
                               learning_rate='invscaling')
    classifier.fit(train_features, train_targets)

    predictions = classifier.predict(test_features)
    score = np.round(metrics.accuracy_score(test_targets, predictions), 2)
    print("Mean accuracy of predictions: " + str(score))


# MLPModel(train_features, test_features, train_targets, test_targets)

# CNN(train_features, test_features, train_targets, test_targets, values, target_data)

quality_values(target_data, values)
