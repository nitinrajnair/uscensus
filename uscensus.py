"""
__author__ = 'NN'

Author : NITINRAJ NAIR

description:
"""

"""You will have to download and install the following 
modules to run the program."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def datamanipulation(file):
    df = pd.read_csv(file, header=None, index_col=False,
                     names=["Age", "Job-Type", "fnlwt", "Education-level", "Years of Experience", \
                            "Marital Status", "Job", "Status", "Race", "Sex", "Capital Gain", \
                            "Capital Loss", "Hrs Per week", "Country", 'Range'], delimiter=",")
    print(file,df['Range'].value_counts())
    df['Range'] = df['Range'].dropna()
    df['Range'] = df['Range'].map({'<=50K': 0, '>50K': 1}).astype(int)

    df['Job'].replace("?", np.nan, inplace=True)
    df['Job'] = df['Job'].dropna()

    df['Job'] = df['Job'].map(
        {'Machine-op-inspct': 1, 'Armed-Forces': 2, 'Handlers-cleaners': 3, 'Craft-repair': 4, 'Sales': 5, \
         'Transport-moving': 6, 'Adm-clerical': 7, 'Other-service': 8, 'Exec-managerial': 9, 'Tech-support': 10, \
         'Protective-serv': 11, 'Farming-fishing': 12, 'Priv-house-serv': 13, 'Prof-specialty': 14})

    # df.groupby(["Job"]).size().plot(kind="bar", fontsize=14)
    # plt.show()

    df['Job-Type'].replace("?", np.nan, inplace=True)
    df['Job-Type'] = df['Job-Type'].dropna()
    df['Job-Type'] = df['Job-Type'].map(
        {'?': -1, 'Without-pay': 0, 'Never-worked': 1, 'Local-gov': 2, 'State-gov': 3, 'Federal-gov': 3,
         'Private': 4, 'Self-emp-not-inc': 5, 'Self-emp-inc': 6})
    # df.groupby(["Job-Type"]).size().plot(kind="bar", fontsize=14)
    # plt.show()

    df = df.drop(columns='fnlwt')
    df = df.drop(columns="Education-level")
    # df["Education-level"] = df["Education-level"].map({'12th':7, '10th':5, '7th-8th':3, 'Assoc-acdm':11, 'Doctorate':16, 'Assoc-voc':12, '11th':6, 'Bachelors':14, '9th':4, 'Prof-school':9,
    #  'Some-college':10, '1st-4th':1, 'HS-grad':13, 'Preschool':8, 'Masters':15, '5th-6th':2})

    # df.groupby(["Education-level"]).size().plot(kind="bar", fontsize=14)
    # plt.show()

    df['Marital Status'] = df['Marital Status'].map(
        {'Married-spouse-absent': 1, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,
         'Never-married': 5, 'Married-AF-spouse': 6})

    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1}).astype(int)

    df['Race'] = df['Race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3,
                                 'Amer-Indian-Eskimo': 4})

    df['Status'] = df['Status'].map({'Unmarried': 0, 'Other-relative': 1, 'Not-in-family': 2,
                                     'Wife': 3, 'Husband': 4, 'Own-child': 5})

    df['Country'].replace("?", np.nan, inplace=True)
    df['Country'] = df['Country'].dropna()
    df['Country'] = df['Country'].map({'Puerto-Rico': 0, 'Haiti': 1, 'Cuba': 2, 'Iran': 3,
                                       'Honduras': 4, 'Jamaica': 5, 'Vietnam': 6, 'Mexico': 7,
                                       'Dominican-Republic': 8,
                                       'Laos': 9, 'Ecuador': 10, 'El-Salvador': 11, 'Cambodia': 12,
                                       'Columbia': 13,
                                       'Guatemala': 14, 'South': 15, 'India': 16, 'Nicaragua': 17,
                                       'Yugoslavia': 18,
                                       'Philippines': 19, 'Thailand': 20, 'Trinadad&Tobago': 21,
                                       'Peru': 22, 'Poland': 23,
                                       'China': 24, 'Hungary': 25, 'Greece': 26, 'Taiwan': 27,
                                       'Italy': 28, 'Portugal': 29,
                                       'France': 30, 'Hong': 31, 'England': 32, 'Scotland': 33,
                                       'Ireland': 34,
                                       'Holand-Netherlands': 35, 'Canada': 36, 'Germany': 37, 'Japan': 38,
                                       'Outlying-US(Guam-USVI-etc)': 39, 'United-States': 40
                                       })
    # hmap = df.corr()
    # plt.subplots(figsize=(12, 9))
    # sns.heatmap(hmap, vmax=.8, annot=True, cmap="BrBG", square=True);
    # plt.show()
    print(df.head)
    x = df.iloc[:, :-1].values.astype(int)

    y = df[['Range']]
    df.to_csv(file+"_out")
    return x,y

def main():
    x_train,y_train=datamanipulation("adult.data")
    x_test, y_test=datamanipulation("adult.test")
    max=0
    for x in range(1000,2000):
        classifier = DecisionTreeClassifier(max_leaf_nodes=x)
        classifier.fit(x_train, y_train)

        y_predict_test = classifier.predict(x_test)
        # accuracy = accuracy_score(y_predict_test,y_test)
        accuracy=classifier.score(x_train,y_train)
        if max<accuracy:
            max=accuracy
        print(accuracy,max)

if __name__ == '__main__':
    main()