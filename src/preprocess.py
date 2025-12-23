import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = ("data/dataset.csv")
def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    x = df.drop("target", axis = 1)
    y = df["target"]
    return train_test_split(x,y,test_size=0.2, random_state=42)
