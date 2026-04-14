import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.model_selection import train_test_split


#==============================================================
# 1.Setup paths
#==============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../data/Diabetes_and_LifeStyle_Dataset_.csv")
train_data_path = os.path.join(script_dir, "../data/train.csv")
test_data_path = os.path.join(script_dir, "../data/test.csv")


#==============================================================
# 2. Fucntion to split the data into train and test sets
#==============================================================

def load_and_split(test_size=0.2, random_state=42):
    raw_data = pd.read_csv(file_path)
    train, test = train_test_split(raw_data, test_size=test_size, random_state=random_state, stratify=raw_data['diabetes_stage'])
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print(f"Saved: train ({len(train)} rows), test ({len(test)} rows)")
    return train, test

if __name__ == '__main__':
    train, test = load_and_split()
