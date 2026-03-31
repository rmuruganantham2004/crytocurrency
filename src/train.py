import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pipeline import DataLoader, FeatureEngineer, PreProcessor, CryptoLSTM, Evaluator

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("Step 1: Data Collection")
    loader = DataLoader(ticker="BTC-USD", start_date="2020-01-01")
    df = loader.fetch_data()

    # EDA: Plot baseline
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title('BTC-USD Historical Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('static/eda_price.png')
    plt.close()
    print("EDA plot saved to static/eda_price.png")

    print("Step 2: Feature Engineering")
    df = FeatureEngineer.add_technical_indicators(df)
    df = df.dropna()

    print("Step 3: Preprocessing")
    # Train-test split (80-20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size - 60:].copy() # include 60 days lookback

    preprocessor = PreProcessor(sequence_length=60)
    X_train, y_train, _ = preprocessor.preprocess_train(train_df)
    
    # Preprocess test manually
    scaled_test = preprocessor.get_scaler().transform(test_df)
    X_test, y_test = [], []
    for i in range(60, len(scaled_test)):
        X_test.append(scaled_test[i - 60:i])
        y_test.append(scaled_test[i, preprocessor.target_idx])
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    print("Step 4: Model Building")
    model = CryptoLSTM(sequence_length=60, features_count=X_train.shape[2])
    
    print("Step 5: Model Training")
    # Setting epochs lower for quicker training on local runs
    model.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    print("Step 6: Predicting on Test Set")
    y_pred = model.predict(X_test)
    
    print("Step 7: Evaluation")
    dates_test = test_df.index[60:]
    y_true_inv, y_pred_inv, rmse, mae = Evaluator.evaluate(
        y_test, y_pred, preprocessor.get_scaler(), preprocessor.target_idx, X_train.shape[2]
    )
    
    Evaluator.plot_predictions(y_true_inv, y_pred_inv, dates_test, save_path='static/prediction_plot.png')
    
    print("Step 8: Saving Artifacts")
    model.save('models/crypto_lstm.pt')
    joblib.dump(preprocessor.get_scaler(), 'models/scaler.pkl')
    joblib.dump(preprocessor.features, 'models/features.pkl')
    print("Training complete. Models and artifacts saved successfully.")

if __name__ == "__main__":
    main()
