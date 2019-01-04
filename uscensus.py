"""
__author__ = 'NN'

Author : NITINRAJ NAIR

description:
"""

"""You will have to downlasd and install the following 
modules to run the program."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df=pd.read_csv("adult.csv",header=None,index_col=False,names=["Age","Job-Type","fnlwt","Education-level","Years of Experience",\
                                                    "Marital Status","Job","Status","Race","Sex","Capital Gain",\
                                                    "Capital Loss","Hrs Per week","Country",'Range'],delimiter="\t")

    # df['Range'] = df['Range'].map({'<=50K': 0, '>50K': 1}).astype(int)
    #
    #
    # df['Job'].replace("?",np.nan,inplace=True)
    # df['Job']=df['Job'].dropna()
    #
    # df['Job']=df['Job'].map({'Machine-op-inspct':1, 'Armed-Forces':2, 'Handlers-cleaners':3, 'Craft-repair':4, 'Sales':5, \
    #                          'Transport-moving':6, 'Adm-clerical':7, 'Other-service':8, 'Exec-managerial':9, 'Tech-support':10, \
    #                          'Protective-serv':11, 'Farming-fishing':12, 'Priv-house-serv':13, 'Prof-specialty':14})
    #
    # df.groupby(["Job"]).size().plot(kind="bar", fontsize=14)
    # plt.show()

    # df['Job-Type'].replace("?",np.nan,inplace=True)
    # df['Job-Type']=df['Job-Type'].dropna()
    #
    # df.groupby(["Job-Type"]).size().plot(kind="bar", fontsize=14)
    # plt.show()
    print(df["fnlwt"].value_counts())
    # df.groupby(["fnlwt"]).size().plot(kind="bar", fontsize=14)
    # plt.show()




if __name__ == '__main__':
    main()