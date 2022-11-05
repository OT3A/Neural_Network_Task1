from random import shuffle
from numpy import NaN
import numpy as np
import pandas as pd
import csv
import pathlib
import cv2
from sympy import true
from calendar import month
from re import X
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.metrics import mean_absolute_error, confusion_matrix, r2_score, accuracy_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression


def sig(x):
    return 1 / (1 + np.exp(-x))

def confusionMatrix(y_test, prediction):
    matrix = np.zeros((2,2))
    original = y_test.tolist()
    for i in range(len(original)):
        if original[i] == 1 and prediction[i] == 1:
            matrix[1,1] += 1 #True True
        elif original[i] == -1 and prediction[i] == 1:
            matrix[0,1] += 1 #False True
        elif original[i] == 1 and prediction[i] == -1:
            matrix[1,0] += 1 #True False
        elif original[i] == -1 and prediction[i] == -1:
            matrix[0,0] += 1 #False False
    matrix = matrix.astype(int)
    return matrix

def initializeWeight(X):
    # return pd.Series(np.random.rand(X))
    x = [np.random.uniform(0.00001, 10**(-20)) for i in range(X)]
    # return pd.Series(np.random.rand(X))
    x = pd.Series(x)
    return x

def train(c1, c2, f1, f2, epochs, eta, bias):
    weights = initializeWeight(3)
    y = c1.iloc[:, 0]
    y = pd.concat([y, c2['species']], axis=0, ignore_index=True)
    y = y.replace([c1.iloc[0,0], c2.iloc[0,0]], [-1,1])
    y = y.astype('int')
    x = c1[[f1, f2]]
    x = pd.concat([x, c2[[f1, f2]]], axis=0, ignore_index=True)
    x0 = np.ones(x.shape[0]) if bias == 1 else np.zeros(x.shape[0])
    x0 = pd.DataFrame(x0, columns = ['bias'])
    x = pd.concat([x0, x], axis=1)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x[:50], y[:50], test_size=0.40, shuffle=True)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x[50:], y[50:], test_size=0.40, shuffle=True)
    x_train = pd.concat([x_train1, x_train2], axis=0, ignore_index=True)
    x_test = pd.concat([x_test1, x_test2], axis=0, ignore_index=True)
    y_train = pd.concat([y_train1, y_train2], axis=0, ignore_index=True)
    y_test = pd.concat([y_test1, y_test2], axis=0, ignore_index=True)
    for i in range(epochs):
        for index, row in x_train.iterrows():
            equation = np.dot(weights, row)
            tempY = round(sig(equation))
            if tempY == 0:
                tempY = -1
            if y_train[index] != tempY:
                diff = y_train[index] - tempY
                addition = eta * diff * row
                weights = weights.add(addition.tolist())
                
    prediction = []
    for index, row in x_test.iterrows():
        prediction.append(round(sig(np.dot(weights, row))))
        # prediction.append(round(np.dot(weights, row)))

    prediction = [-1 if i == 0 else i for i in prediction]
    print(f'prediction = {prediction}\ny test     = {y_test.tolist()}')
    print(f'r2 accuracy = {r2_score(y_test, prediction)}')
    print(f'accuracy = {accuracy_score(y_test, prediction)}')

    builtcm = confusion_matrix(y_test, prediction)
    cm = confusionMatrix(y_test, prediction)
    print(f'confusion_matrix = {builtcm}')
    print(f'our confusion_matrix = {cm}')
    plt.figure(figsize = (10,8))
    sns.heatmap(cm, annot=True, cmap= 'flare',  fmt='d', cbar=True)
    plt.xlabel('Predicted_Label')
    plt.ylabel('Truth_Label')
    plt.title('Confusion Matrix')
    plt.show()

    xmin = x[f1].min()
    xmax = x[f1].max()
    xline = [xmin, xmax]
    line = []
    line.append(- (weights[1] * xline[0] + weights[0]) / weights[2] if bias == 1 else - (weights[1] * xline[0]) / weights[2])
    line.append(- (weights[1] * xline[1] + weights[0]) / weights[2] if bias == 1 else - (weights[1] * xline[1]) / weights[2])
    # print(f'line = {line}')
    # print(f'xline = {xline}')
    # print(f'weight = {weights[1]}')
    # line = []
    # for _, row in x.iterrows():
    #     # line.append(np.array((np.dot(weights, row))))
    #     line.append(round(sig(np.dot(weights, row))))

    # line = [-1 if i == 0 else i for i in line]
    # xline = [i for i in range(x.shape[0])]
    
    plt.scatter(c1[[f1]], c1[[f2]])
    plt.scatter(c2[[f1]], c2[[f2]])
    plt.plot(xline, line)
    plt.show()

    return weights, accuracy_score(y_test, prediction)

def main(class1, class2, feature1, feature2, epochs=1000, eta=0.05, bias=0):
    data = pd.read_csv('penguins.csv')
    i = data['gender'].value_counts()

    data['gender'] = data['gender'].replace(['male', 'female', NaN], [0, 1, 0])
    data['gender'] = data['gender'].astype('int')

    c1 = data.iloc[:50, :]
    c2 = data.iloc[50:100, :]
    c3 = data.iloc[100:, :]
    plt.scatter(c1[[feature1]], c1[[feature2]])
    plt.scatter(c2[[feature1]], c2[[feature2]])
    plt.scatter(c3[[feature1]], c3[[feature2]])
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

    # print(c1)
    # print(c2)
    # print(c3)

    # weights = initializeWeight(3)
    # print(f'weights before train = \n{weights}')
    # print(f'weights after train = \n{weights}')
    # print(data.loc[data['species'] == 'Chinstrap'])

    weights, accuracy = train(data.loc[data['species'] == class1], data.loc[data['species'] == class2], feature1, feature2, epochs, eta, bias)
    print(f'weights = \n{weights}\naccuracy = {accuracy}')
    # with open('accuracy.txt', 'a') as f:
    #     f.write(f'classes ({class1}, {class2}), features ({feature1}, {feature2}), accuracy ({accuracy})\n')
    return accuracy

if __name__ == '__main__':
    main('Adelie', 'Chinstrap', 'bill_depth_mm', 'bill_length_mm')