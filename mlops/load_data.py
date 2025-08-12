import pandas as pd

def load_data():
    df = pd.read_parquet("data/rides.parquet")
    print("Data loaded from Parquet:")
    print(df)

if __name__ == "__main__":
    load_data()
