import pandas as pd

def csv_to_parquet():
    df = pd.read_csv("data/rides.csv")
    df.to_parquet("data/rides.parquet")
    print("Converted rides.csv to rides.parquet")

if __name__ == "__main__":
    csv_to_parquet()
