import numpy as np
import pandas
from sklearn.model_selection import train_test_split

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''


def getData():
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge'] = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',', '', regex=True))
    dataset_1['Manhattan Bridge'] = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',', '', regex=True))
    dataset_1['Queensboro Bridge'] = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',', '', regex=True))
    dataset_1['Williamsburg Bridge'] = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',', '', regex=True))
    dataset_1['Williamsburg Bridge'] = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',', '', regex=True))
    print(dataset_1.to_string())  # This line will print out your data
    return dataset_1


# brooklyn = getData().loc[:, "Brooklyn Bridge"]
brooklyn = list(getData()["Brooklyn Bridge"])
manhattan = list(getData()["Manhattan Bridge"])
williamsburg = list(getData()["Williamsburg Bridge"])
queensboro = list(getData()["Queensboro Bridge"])
totalTraffic = list(getData()["Total"])
precipitation = list(getData()["Precipitation"])

X = np.array([brooklyn, manhattan, williamsburg, queensboro])
Y = np.array(totalTraffic)


def normalize_train(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    train = (X_train - mean) / std

    return train, mean, std


def normalize_test(X_test, trn_mean, trn_std):
    test = (X_test - trn_mean) / trn_std
    return test


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)
[X_train, trn_mean, trn_std] = normalize_train(X_train)
X_test = normalize_test(X_test, trn_mean, trn_std)

# getData()
# print(brooklyn)
# print(manhattan)
# print(williamsburg)
# print(queensboro)
