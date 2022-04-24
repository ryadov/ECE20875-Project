import math
import numpy as np
import seaborn as sns
import statsmodels
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from MiniProjectPath1 import getData

# def range_with_floats(start, stop, step):
#     while stop > start:
#         yield start
#         start += step

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def main():
    totalTraffic = list(getData()["Total"])
    precipitation = list(getData()["Precipitation"])
    precipitation_threshold = 0.098
    testing_size = 0.2
    raining = [1 if p > precipitation_threshold else 0 for p in precipitation]
    plt.scatter(totalTraffic, raining, marker='D', c='blue', label= 'Raining State')
    # plt.show()    #uncomment to view plot of precipitation against number of bicyclist


    X_train, X_test, y_train, y_test = train_test_split(totalTraffic, raining, test_size=testing_size, shuffle=False)
    model = LogisticRegression()
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    model.fit(X_train, y_train)
    # print(model.predict_proba(X_test))
    print(f"The score of this model is {model.score(X_test, y_test):.3f}")
    # plt.scatter(X_train, y_train, marker='o', color="black")
    x_cont = np.array(np.linspace(min(totalTraffic),max(totalTraffic), 1000)).reshape(-1, 1)
    # print(model.intercept_)
    sig = sigmoid(sorted(x_cont) * model.coef_ + model.intercept_)
    # plt.scatter(x_cont, sig, color="red", linewidth=3)
    sns.regplot(x=totalTraffic, y=raining, logistic=True, ci=None, label='Probability of Rain')
    plt.title('Raining Prediction against Total Traffic')
    plt.xlabel('Total Traffic')
    plt.ylabel('Raining Or Not Raining')
    plt.legend(loc="center right")
    plt.show()


    return

if __name__ == "__main__":
    main()
