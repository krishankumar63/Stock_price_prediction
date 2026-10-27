import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.20
CSV_FILE = "stock_dataset.csv"
MODEL_FILE = "trained_models.pkl"


def calculate_mse(y_true, y_pred):
    """Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def calculate_r2(y_true, y_pred):
    """R-squared Score"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# SECTION 1: EDA & PREPROCESSING 

def load_dataset():
    """Loads the static dataset from stock_dataset.csv."""
    df = pd.read_csv(CSV_FILE)
    print("Dataset Shape:", df.shape)
    return df


def preprocess_data(df):
    """
    Performs all Data Cleaning, EDA, IQR Outlier Handling,
    Train/Test Split, and StandardScaler in one place.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA CLEANING, EDA AND PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]
    target_column = "Target_Next_Close"

    # 1. Check Missing Values Before Cleaning
    print("\n--- Missing Values Before Cleaning ---")
    print(df[features + [target_column]].isnull().sum())

    # 2. Check and Remove Duplicate Rows
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicate rows removed.")
    else:
        print("No duplicate rows found.")

    # 3. Fill Missing Values with Median
    for column in features + [target_column]:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            print(f"Missing values in {column} filled with median: {median_value:.2f}")

    print("\n--- Missing Values After Cleaning ---")
    print(df[features + [target_column]].isnull().sum())

    # 4. Statistical Summary (EDA)
    eda_summary = df[features].describe().round(2)
    print("\n--- Statistical Summary ---")
    print(eda_summary)

    # 5. Outlier Detection and Handling using IQR
    print("\n--- Outlier Detection and Handling (IQR) ---")
    for column in features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outliers.sum()
        print(f"{column}: {outlier_count} outliers detected")
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # 6. Feature Correlation with Target
    correlation_matrix = df[features + [target_column]].corr()
    print("\n--- Correlation with Target_Next_Close ---")
    print(correlation_matrix )
    # print(correlation_matrix[target_column].drop(target_column).round(4))

    # 7. Separate Input Features (X) and Target (y)
    X = df[features]
    y = df[target_column]
    print("\nInput features shape:", X.shape)
    print("Output variable shape:", y.shape)

    # 8. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. Train/Test Split AFTER Scaling (80% Train, 20% Test)
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Training samples:", X_train_scaled.shape[0])
    print("Testing samples:", X_test_scaled.shape[0])

    return eda_summary, scaler, X_train_scaled, X_test_scaled, y_train, y_test



# SECTION 2: LINEAR REGRESSION, RANDOM FOREST & GRIDSEARCHCV OPTIMIZATION

def train_and_save_models():
    """
    Runs Preprocessing once, then trains Linear Regression,
    Base Random Forest, and GridSearchCV Optimized Random Forest.
    """
    # 1. Run all preprocessing once
    df = load_dataset()

    (eda_summary, scaler,
     X_train_scaled, X_test_scaled, y_train, y_test) = preprocess_data(df)

    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING AND GRIDSEARCHCV OPTIMIZATION")
    print("=" * 60)

    # --- Model 1: Linear Regression ---
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    print("Linear Regression training completed.")

    # --- Model 2: Default Random Forest Regressor ---
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_base.fit(X_train_scaled, y_train)
    rf_base_preds = rf_base.predict(X_test_scaled)
    print("Random Forest (Default) training completed.")

    # --- Model 3: Hyperparameter Tuning with GridSearchCV  ---
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    rf_opt = grid_search.best_estimator_
    rf_opt_preds = rf_opt.predict(X_test_scaled)
    print("Random Forest (GridSearchCV) optimization completed.")
    print("Best Parameters:", grid_search.best_params_)

    # --- Evaluate All Models Using Assignment 6 Metric Functions ---
    metrics = {
        "Linear Regression ": {
            "MSE": round(calculate_mse(y_test, lr_preds), 2),
            "MAE": round(calculate_mae(y_test, lr_preds), 2),
            "R2 Score": round(calculate_r2(y_test, lr_preds), 4)
        },
        "Random Forest - Base ": {
            "MSE": round(calculate_mse(y_test, rf_base_preds), 2),
            "MAE": round(calculate_mae(y_test, rf_base_preds), 2),
            "R2 Score": round(calculate_r2(y_test, rf_base_preds), 4)
        },
        "Random Forest - GridSearchCV Tuned ": {
            "MSE": round(calculate_mse(y_test, rf_opt_preds), 2),
            "MAE": round(calculate_mae(y_test, rf_opt_preds), 2),
            "R2 Score": round(calculate_r2(y_test, rf_opt_preds), 4)
        }
    }

    print("\n--- Model Performance Comparison ---")
    print(pd.DataFrame(metrics).T)

    # Save trained objects for the unchanged Streamlit UI
    bundle = {
        "lr_model": lr_model,
        "rf_base": rf_base,
        "rf_opt": rf_opt,
        "scaler": scaler,
        "metrics": metrics,
        "best_params": grid_search.best_params_,
        "eda_summary": eda_summary,
        "y_test": y_test.values[:100],
        "lr_preds": lr_preds[:100],
        "rf_opt_preds": rf_opt_preds[:100]
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)

    return bundle



# SECTION 3: MODEL LOADING & PREDICTION FUNCTIONS FOR UI

def load_trained_bundle():
    """Loads pre-trained models from disk without retraining."""
    if not os.path.exists(MODEL_FILE):
        return train_and_save_models()
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict_any_stock(ticker):
    """Predicts next day closing price for a selected stock using pre-trained models."""
    bundle = load_trained_bundle()
    features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]

    # fetching stock data for the last 3 months using yfinance , todays data
    stock_df = yf.download(ticker, period="3mo", progress=False)
    if stock_df.empty or len(stock_df) < 12:
        return None          
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    stock_df["MA5"] = stock_df["Close"].rolling(window=5).mean()
    stock_df["MA10"] = stock_df["Close"].rolling(window=10).mean()
    stock_df = stock_df.dropna()
                

    latest = stock_df.iloc[-1]
    close_val = float(latest["Close"])

    # Create input DataFrame with the 7 features 
    X_input = pd.DataFrame([[
        float(latest["Open"]),
        float(latest["High"]),
        float(latest["Low"]),
        close_val,
        float(latest["Volume"]),
        float(latest["MA5"]),
        float(latest["MA10"])
    ]], columns=features)

    # Scale input and predict
    X_scaled = bundle["scaler"].transform(X_input)
    pred_lr = float(bundle["lr_model"].predict(X_scaled)[0])
    pred_rf = float(bundle["rf_opt"].predict(X_scaled)[0])

    return {
        "history_df": stock_df.tail(60),
        "latest_row": X_input.iloc[0],
        "last_close": close_val,
        "pred_lr": pred_lr,
        "pred_rf": pred_rf
    }


def predict_from_manual_input(open_p, high_p, low_p, close_p, volume, ma5, ma10):
    """Predicts next day closing price from manual user inputs."""
    bundle = load_trained_bundle()
    features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]

    X_input = pd.DataFrame(
        [[open_p, high_p, low_p, close_p, volume, ma5, ma10]],
        columns=features
    )

    X_scaled = bundle["scaler"].transform(X_input)
    pred_lr = float(bundle["lr_model"].predict(X_scaled)[0])
    pred_rf = float(bundle["rf_opt"].predict(X_scaled)[0])

    return pred_lr, pred_rf


if __name__ == "__main__":
    train_and_save_models()