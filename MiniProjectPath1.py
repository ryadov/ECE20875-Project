from numpy import cov
from scipy.stats import pearsonr
import numpy as np
import pandas
from matplotlib import pyplot as plt
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
    # print(dataset_1.to_string())  # This line will print out your data
    return dataset_1

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

def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    B = np.linalg.lstsq(X, y, rcond=None)
    return B[0]


def main():
    # brooklyn = getData().loc[:, "Brooklyn Bridge"]
    brooklyn = list(getData()["Brooklyn Bridge"])
    manhattan = list(getData()["Manhattan Bridge"])
    williamsburg = list(getData()["Williamsburg Bridge"])
    queensboro = list(getData()["Queensboro Bridge"])
    totalTraffic = list(getData()["Total"])
    tempLow = list(getData()["Low Temp"])
    tempHigh = list(getData()["High Temp"])
    precipitation = list(getData()["Precipitation"])

    plt.scatter(brooklyn, totalTraffic, label="Brooklyn")
    plt.scatter(manhattan, totalTraffic, label="Manhattan")
    plt.scatter(williamsburg, totalTraffic, label="Williamsburg")
    plt.scatter(queensboro, totalTraffic, label="Queensboro")
    plt.title('Each Bridge\'s Contribution to the Total Traffic')
    plt.xlabel('Bridge Traffic')
    plt.ylabel('Total Traffic')
    plt.legend(loc="upper left")
    plt.show()

    X = np.array([brooklyn, manhattan, williamsburg, queensboro]).T
    Y = np.array(totalTraffic).T
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
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
    plt.plot(lmbda, MSE)
    plt.title('Mean-Squared Error as a function of $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.show()
    print(
        "Best lambda tested is "
        + str(lmda_best)
        + ", which yields an MSE of "
        + str(MSE_best)
    )
    print(
        f"The equation is y = {model_best.coef_[0]:.0f}x1 + {model_best.coef_[1]:.0f}x2 + {model_best.coef_[2]:.0f}x3 + {model_best.coef_[3]:.0f}x4 + {model_best.intercept_:.0f}"
        f"\nwhere x1 = brooklyn, x2 = manhattan, x3 = williamsburg and x4 = queensboro")
    print(f"The bridge that should NOT have a sensor installed on is Brooklyn, due to having the lowest coefficient, {min(model_best.coef_):.0f}, out of {model_best.coef_.round()}")
    # getData()
    # print(brooklyn)
    # print(manhattan)
    # print(williamsburg)
    # print(queensboro)
    ####################PART TWO STARTS HERE#####################

    # X2 = np.array([tempHigh, tempLow, precipitation]).T
    # Y2 = totalTraffic
    plt.scatter(tempHigh, totalTraffic, label="Temperature High")
    plt.scatter(tempLow, totalTraffic, label="Temperature Low")
    plt.title('Temperature Measurements against Total Traffic')
    plt.xlabel('Temperature')
    plt.ylabel('Total Traffic')
    plt.legend(loc="upper left")
    plt.show()

    plt.scatter(precipitation, totalTraffic, label="Precipitation")
    plt.title('Precipitation Measurements against Total Traffic')
    plt.xlabel('Precipitation')
    plt.ylabel('Total Traffic')
    plt.legend(loc="upper left")
    plt.show()

    degrees = [1, 2, 3, 4, 5]
    paramFits = []
    # for i in degrees:
    #     paramFits.append(feature_matrix(data, i))
        # paramFits.append(least_squares(X2, Y2))
    Xtemp = (np.array(tempLow) + np.array(tempHigh))/2
    Xtemp = Xtemp.T  # horizontal array
    # Xf = Xtemp[0]  # horizontal array one-dimensional
    Xf2 = np.sort(Xtemp)
    Xf = feature_matrix(Xtemp, 2)
    Xf = Xf.T  # final X vertical

    Ytemp = np.array(totalTraffic).T
    beta = least_squares(Xf, Y)  # calculating beta
    Ybar = Xf @ beta
    temp = []
    for y in Ybar:
        temp.append(y)
    temp.sort()
    plt.scatter(tempHigh, totalTraffic, label="Temperature High Traffic")
    plt.scatter(tempLow, totalTraffic, label="Temperature Low Traffic")
    plt.title('Temperature Measurements against Total Traffic')
    plt.xlabel('Temperature')
    plt.ylabel('Total Traffic')
    plt.plot(Xf2, np.array(temp), c='black', label=f'Predicted Traffic')
    plt.legend(loc="upper left")
    plt.show()
    MSE = []
    ind = 0
    for d in degrees:
        Xtemp = np.array(precipitation)
        Xtemp = Xtemp.T  # horizontal array
        Xf2 = np.sort(Xtemp)[::-1]
        Xf = feature_matrix(Xtemp, d)
        Xf = Xf.T  # final X vertical
        beta = least_squares(Xf, Ytemp)  # calculating beta
        Ybar = Xf @ beta
        MSE.append(((Ytemp - Ybar) ** 2).mean())
        print(f"For degree = {d}, beta = {beta}")
        print(f"For degree = {d}, MSE = {MSE[ind]} ")

        temp = []
        for y in Ybar:
            temp.append(y)
        temp.sort()
        plt.plot(Xf2, np.array(temp), label=f'd = {d}')
        paramFits.append(beta)
        ind += 1
    plt.scatter(precipitation, Ytemp, label="Precipitation Traffic")
    plt.title('Precipitation Measurements against Total Traffic')
    plt.xlabel('Precipitation')
    plt.ylabel('Total Traffic')
    plt.plot(Xf2, np.array(temp), c='black', label=f'Predicted Traffic')
    plt.legend(loc="upper left")
    plt.show()
    corr, _ = pearsonr(precipitation, totalTraffic)
    print(f"The best model is with degree {MSE.index(min(MSE))} with an MSE of {min(MSE)}")
    # Correlation = cov(precipitation, totalTraffic)/(np.std(precipitation) * np.std(totalTraffic))
    print(f"Correlation of Precipitation to Total Traffic is {corr:.3f}, which indicates a strong negative correlation.")
    return

def feature_matrix(x, d):
    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    power = 2
    X = x.copy()
    while power <= d:
        temp = np.array([i ** power for i in x])
        X = np.vstack((temp, X))
        power += 1
    xlen = len(x)  # length of array
    Ones = xlen * [1]  # xlen ones in horizontal list
    OnesArr = np.array(Ones)  # ones in horizontal array
    X = np.vstack((X, OnesArr))  # appending ones as the last row
    return X

if __name__ == "__main__":
    main()