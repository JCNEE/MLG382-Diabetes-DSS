import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split


#==============================================================
# 1.Setup paths
#==============================================================

os.makedirs("artifacts", exist_ok=True) 
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../data/Diabetes_and_LifeStyle_Dataset_.csv")
train_data_path = os.path.join(script_dir, "../data/train.csv")
test_data_path = os.path.join(script_dir, "../data/test.csv")


#==============================================================
# 2. Function to split the data into train and test sets
#==============================================================

def load_and_split(test_size=0.2, random_state=42):
    raw_data = pd.read_csv(file_path)
    train, test = train_test_split(raw_data, test_size=test_size, random_state=random_state, stratify=raw_data['diabetes_stage'])
    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    print(f"Saved: train ({len(train)} rows), test ({len(test)} rows)")
    return train, test



#==============================================================
# 3. Function to visualize the distribution of diabetes stages in the train and test sets
#==============================================================

def plot_eda(df): 
   

    # Target distribution
    df["diabetes_stage"].value_counts().plot(kind="bar") 
    plt.title("Diabetes Stage Distribution") 
    plt.tight_layout() 
    plt.savefig("artifacts/target_distribution.png") 
    plt.close() 

    # Correlation heatmap
    num_df = df.select_dtypes(include="number") 
    plt.figure(figsize=(14, 10)) 
    sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False) 
    plt.tight_layout() 
    plt.savefig("artifacts/correlation_heatmap.png") 
    plt.close() 

if __name__ == '__main__':
    train, test = load_and_split()

    full = pd.read_csv(file_path) 
    plot_eda(full)