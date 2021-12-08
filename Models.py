import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from random import randint, uniform
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Read in csv file and slpit by ;
wine_df = pd.read_csv('winequality-red.csv', sep=";")

# Create target dataframe of the quality column
target_data = wine_df['quality']
del wine_df['quality']
# Create dataframe of all other values
values = wine_df

train_features, test_features, train_targets, test_targets = train_test_split(values, target_data, test_size=0.2)


def quality_values(target_data, values):
    '''
    This function takes in the target data amd tjhe values and produces a scatter graph comparing the two,
    change the y value for different values
    :param target_data:
    :param values:
    '''

    x = target_data
    y = values['alcohol']

    # Normalized data to allow for more accurate gradients
    y_max = y.max()
    normalized_y = y / y_max

    plt.scatter(x, normalized_y)

    # Create a trend-line
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

def CNN(train_features, test_features, train_targets, test_targets):
    '''
    Implements a CNN, trains it and returns it
    :param train_features:
    :param test_features:
    :param train_targets:
    :param test_targets:
    :return: model: A Tensorflow Sequential Model of the CNN
    '''
    classes_num = 10

    activation = 'relu'

    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(np.prod(train_features.shape[1:]),)))
    model.add(Dense(512, activation=activation))
    model.add(Dense(classes_num, activation='softmax'))

    # rmsprop
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_features, train_targets, batch_size=512, epochs=250,
                        verbose=1, validation_data=(test_features, test_targets))

    [test_loss, test_acc] = model.evaluate(test_features, test_targets)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

    # Loss and accuracy graphs, uncomment to generate

    # # Plot the Loss Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(history.history['loss'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    # plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.title('Loss Curves', fontsize=16)
    #
    # plt.show()
    #
    # # Plot the Accuracy Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    # plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Accuracy Curves', fontsize=16)
    #
    # plt.show()

    return model


def MLPModel(train_features, test_features, train_targets, test_targets):
    """
    Implements a MLP, trains it then returns it
    :param train_features:
    :param test_features:
    :param train_targets:
    :param test_targets:
    :return: model: The MLP model
    """
    all_means = 0
    iterations = 50
    for i in range(iterations):
        model = MLPClassifier(hidden_layer_sizes=12, max_iter=1000, activation='relu', solver='adam', verbose=10,
                              learning_rate='adaptive')
        model.fit(train_features, train_targets)

        predictions = model.predict(test_features)
        score = np.round(metrics.accuracy_score(test_targets, predictions), 2)
        print("Mean accuracy of predictions: " + str(score))
        all_means += score
    print("Average means = " + str(all_means / iterations))

    return model


def generateWines(values):
    """
    This function generates a dataframe of randomly generated wines, without quality measures. The range of each
    feature is generated by taking the difference between max and min, then subtracting it from the min, to a lower
    limit of 0 and adding it to the max.
    :param values:
    :return: newWines:
    """
    numOfWinesToGenerate = 100
    newWines = pd.DataFrame()

    minFixedAcidity = values['fixed acidity'].min()
    maxFixedAcidity = values['fixed acidity'].max()
    fixedAcidityDifference = maxFixedAcidity - minFixedAcidity
    fixedAcidityRange = [
        minFixedAcidity - fixedAcidityDifference if minFixedAcidity - fixedAcidityDifference > 0 else 0,
        maxFixedAcidity + fixedAcidityDifference]

    minVolatileAcidity = values['volatile acidity'].min()
    maxVolatileAcidity = values['volatile acidity'].max()
    volatileAcidityDifference = maxVolatileAcidity - minVolatileAcidity
    volatileAcidityRange = [
        minVolatileAcidity - volatileAcidityDifference if minVolatileAcidity - minVolatileAcidity > 0 else 0,
        maxVolatileAcidity + volatileAcidityDifference]

    minCitricAcid = values['citric acid'].min()
    maxCitricAcid = values['citric acid'].max()
    citricAcidDifference = maxCitricAcid - minCitricAcid
    citricAcidRange = [minCitricAcid - citricAcidDifference if minCitricAcid - citricAcidDifference > 0 else 0,
                       maxCitricAcid + citricAcidDifference]

    minResidualSugar = values['residual sugar'].min()
    maxResidualSugar = values['residual sugar'].max()
    residualSugarDifference = maxResidualSugar - minResidualSugar
    residualSugarRange = [
        minResidualSugar - residualSugarDifference if minResidualSugar - residualSugarDifference > 0 else 0,
        maxResidualSugar + residualSugarDifference]

    minChlorides = values['chlorides'].min()
    maxChlorides = values['chlorides'].max()
    chloridesDifference = maxChlorides - minChlorides
    chloridesRange = [minChlorides - chloridesDifference if minChlorides - chloridesDifference > 0 else 0,
                      maxChlorides + chloridesDifference]

    minFreeSulphurDioxide = values['free sulfur dioxide'].min()
    maxFreeSulphurDioxide = values['free sulfur dioxide'].max()
    freeSulphurDioxideDifference = maxFreeSulphurDioxide - minFreeSulphurDioxide
    freeSulphurDioxideRange = [
        minFreeSulphurDioxide - freeSulphurDioxideDifference if
        minFreeSulphurDioxide - freeSulphurDioxideDifference > 0 else 0,
        maxFreeSulphurDioxide + freeSulphurDioxideDifference]

    minTotalSulphurDioxide = values['total sulfur dioxide'].min()
    maxTotalSulphurDioxide = values['total sulfur dioxide'].max()
    totalSulphurDioxideDifference = maxTotalSulphurDioxide - minTotalSulphurDioxide
    totalSulphurDioxideRange = [
        minTotalSulphurDioxide - totalSulphurDioxideDifference if
        minTotalSulphurDioxide - totalSulphurDioxideDifference > 0 else 0,
        maxTotalSulphurDioxide + totalSulphurDioxideDifference]

    minDensity = values['density'].min()
    maxDensity = values['density'].max()
    densityDifference = maxDensity - minDensity
    densityRange = [minDensity - densityDifference if minDensity - densityDifference > 0 else 0,
                    maxDensity + densityDifference]

    minpH = values['pH'].min()
    maxpH = values['pH'].max()
    phDifference = maxpH - minpH
    pHRange = [minpH - phDifference if minpH - phDifference > 0 else 0, maxpH + phDifference]

    minSulphates = values['sulphates'].min()
    maxSulphates = values['sulphates'].max()
    sulphatesDifference = maxSulphates - minSulphates
    sulphatesRange = [minSulphates - sulphatesDifference if minSulphates - sulphatesDifference > 0 else 0,
                      maxSulphates + sulphatesDifference]

    minAlcohol = values['alcohol'].min()
    maxAlcohol = values['alcohol'].max()
    alcoholDifference = maxAlcohol - minAlcohol
    alcoholRange = [minAlcohol - alcoholDifference if minAlcohol - alcoholDifference > 0 else 0,
                    maxAlcohol + alcoholDifference]

    for i in range(numOfWinesToGenerate):
        newWines = newWines.append([[uniform(fixedAcidityRange[0], fixedAcidityRange[1]),
                                     uniform(volatileAcidityRange[0], volatileAcidityRange[1]),
                                     uniform(citricAcidRange[0], citricAcidRange[1]),
                                     uniform(residualSugarRange[0], residualSugarRange[1]),
                                     uniform(chloridesRange[0], chloridesRange[1]),
                                     randint(freeSulphurDioxideRange[0], freeSulphurDioxideRange[1]),
                                     randint(totalSulphurDioxideRange[0], totalSulphurDioxideRange[1]),
                                     uniform(densityRange[0], densityRange[1]),
                                     uniform(pHRange[0], pHRange[1]),
                                     uniform(sulphatesRange[0], sulphatesRange[1]),
                                     (uniform(alcoholRange[0], alcoholRange[1]))]], ignore_index=True)
    return newWines


def predictQuality(newWines, MLPModel, CNNModel):
    """
    A simple function that just decides which model to use, it returns an array of the predictions
    :param newWines:
    :param MLPModel:
    :param CNNModel:
    :return: predictions: Array of the predictions
    """
    if MLPModel is not None:
        predictions = MLPModel.predict(newWines)
    else:
        predictions = CNNModel.predict_on_batch(newWines)

    return predictions


mode = 0
if mode == 0:
    prediction = predictQuality(generateWines(values),
                                MLPModel(train_features, test_features, train_targets, test_targets),
                                None)
else:
    prediction = predictQuality(generateWines(values),
                                None,
                                CNN(train_features, test_features, train_targets, test_targets))

# A histogram of the predictions generated by the model
plt.hist(prediction, bins=10, histtype='stepfilled')
plt.title('Histogram of 100  New Wines')
plt.xlabel('Predicted Quality')
plt.ylabel('Number of Wines')
plt.show()
