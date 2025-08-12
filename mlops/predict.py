import joblib
import pandas as pd

def predict(new_ride_ids):
    # Load the saved model
    model = joblib.load("model.joblib")

    # Prepare input data as DataFrame
    X_new = pd.DataFrame({"ride_id": new_ride_ids})

    # Make predictions
    preds = model.predict(X_new)

    # Print predictions
    for ride_id, pred in zip(new_ride_ids, preds):
        print(f"Ride ID {ride_id} — predicted duration: {pred:.2f} minutes")

if __name__ == "__main__":
    # Example prediction for ride IDs 5, 6, and 7
    predict([5, 6, 7])
