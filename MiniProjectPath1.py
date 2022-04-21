import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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
    dataset_1['Total'] = pandas.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
    dataset_1['Precipitation'] = pandas.to_numeric(dataset_1['Precipitation'].replace(',', '', regex=True))
    #print(dataset_1.to_string())  # This line will print out your data
    return dataset_1


# brooklyn = getData().loc[:, "Brooklyn Bridge"]
brooklyn = list(getData()["Brooklyn Bridge"])
manhattan = list(getData()["Manhattan Bridge"])
williamsburg = list(getData()["Williamsburg Bridge"])
queensboro = list(getData()["Queensboro Bridge"])
totalTraffic = list(getData()["Total"])
precipitation = list(getData()["Precipitation"])
#print(brooklyn)
X = np.array([brooklyn, manhattan, williamsburg, queensboro]).T
Y = np.array(totalTraffic).T


def normalize_train(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    train = (X_train - mean) / std

    return train, mean, std


def normalize_test(X_test, trn_mean, trn_std):
    test = (X_test - trn_mean) / trn_std
    return test


def get_lambda_range():
    lmbda = np.logspace(-1, 3, num=51)
    return lmbda


def train_model(X, y, l):
    model = Ridge(alpha=l, fit_intercept=True)
    model.fit(X, y)
    return model


def error(X, y, model):
    A = model.predict(X)
    mse = ((y - A) ** 2).mean()
    return mse


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)
[X_train, trn_mean, trn_std] = normalize_train(X_train)
X_test = normalize_test(X_test, trn_mean, trn_std)
lmbda = get_lambda_range()

MODEL = []
MSE = []

for l in lmbda:
    model = train_model(X_train, y_train, l)
    mse = error(X_test, y_test, model)
    MODEL.append(model)
    MSE.append(mse)

i = np.argmin(MSE)
[lmda_best, MSE_best, model_best] = [lmbda[i], MSE[i], MODEL[i]]

print(
    "Best lambda tested is "
    + str(lmda_best)
    + ", which yields an MSE of "
    + str(MSE_best)
)
print(f"The equation is {model_best.coef_[0]} x1 + {model_best.coef_[1]} x2 + {model_best.coef_[2]} x3 + {model_best.coef_[3]} x4 \n where x1 = brooklyn, x2 = manhattan, x3 = williamsburg and x4 = queensboro")
# getData()
# print(brooklyn)
# print(manhattan)
# print(williamsburg)
# print(queensboro)
