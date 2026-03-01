import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data(filepath):
    ev_charge_df = pd.read_csv(filepath)
    
    # Null imputation as per Colab snippet
    null_cols = ['Energy Consumed (kWh)', 'Charging Rate (kW)', 'Distance Driven (since last charge) (km)']
    for col in null_cols:
        if col in ev_charge_df.columns:
            ev_charge_df[col] = ev_charge_df[col].fillna(ev_charge_df[col].median())
        
    # Datetime Extraction
    ev_charge_df['Charging Start Time'] = pd.to_datetime(ev_charge_df['Charging Start Time'])
    ev_charge_df['Charging End Time'] = pd.to_datetime(ev_charge_df['Charging End Time'])
    ev_charge_df['Hour'] = ev_charge_df['Charging Start Time'].dt.hour
    ev_charge_df['DayOfWeek'] = ev_charge_df['Charging Start Time'].dt.dayofweek
    ev_charge_df['Month'] = ev_charge_df['Charging Start Time'].dt.month
    ev_charge_df['IsWeekend'] = ev_charge_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Grouping Data to calculate Macroscopic Total Demand
    demand_df = ev_charge_df.groupby(['Charging Station Location', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']).agg({
        'Energy Consumed (kWh)': 'sum',
        'Temperature (°C)': 'mean',
        'Battery Capacity (kWh)': 'mean'
    }).reset_index()
    
    demand_df.rename(columns={'Energy Consumed (kWh)': 'Total_Demand'}, inplace=True)
    
    # Encoding Location
    le = LabelEncoder()
    demand_df['Location_Encoded'] = le.fit_transform(demand_df['Charging Station Location'])
    
    # Save the encoder for the simulation tool
    joblib.dump(le, "location_encoder.pkl")
    
    return demand_df, ev_charge_df

def build_and_train_model(demand_df):
    X = demand_df[['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Temperature (°C)', 'Battery Capacity (kWh)', 'Location_Encoded']]
    y = demand_df['Total_Demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using the exact model and parameters requested
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, rf_preds)
    rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    r2 = r2_score(y_test, rf_preds)
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    # Generate Actual vs Predicted DataFrame for visualization
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': rf_preds}).head(50).reset_index(drop=True)
    
    return rf_model, metrics, results_df

def save_model(model, filename='ev_demand_model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='ev_demand_model.pkl'):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

def predict_demand(hour, day_of_week, month, is_weekend, temp, battery_cap, location):
    model = load_model("ev_demand_model.pkl")
    le = joblib.load("location_encoder.pkl")
    
    loc_encoded = le.transform([location])[0]
    
    input_data = pd.DataFrame({
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'Month': [month],
        'IsWeekend': [is_weekend],
        'Temperature (°C)': [temp],
        'Battery Capacity (kWh)': [battery_cap],
        'Location_Encoded': [loc_encoded]
    })
    
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    filepath = "ev_charging_patterns.csv"
    if os.path.exists(filepath):
        print(f"Loading and processing data from {filepath}...")
        demand_df, _ = load_and_preprocess_data(filepath)
        
        print("Training Macroscopic Demand Model...")
        model, metrics, results_df = build_and_train_model(demand_df)
        
        print("\nRandom Forest Metrics ---")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R-Squared: {metrics['R2']:.2f}\n")
        
        save_model(model, "ev_demand_model.pkl")
        print("Model and Encoders saved successfully.")
    else:
        print(f"Error: {filepath} not found.")