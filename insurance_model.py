import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Sample features that might be relevant for vehicle insurance pricing
def create_insurance_model():
    # Sample data structure (you'll need to replace this with your actual data)
    data = {
        'age': np.random.randint(18, 80, 1000),
        'vehicle_age': np.random.randint(0, 20, 1000),
        'vehicle_value': np.random.uniform(5000, 100000, 1000),
        'annual_mileage': np.random.uniform(1000, 50000, 1000),
        'accident_history': np.random.randint(0, 3, 1000),
        'insurance_premium': np.random.uniform(500, 5000, 1000)  # Target variable
    }
    
    df = pd.DataFrame(data)
    
    # Separate features and target
    X = df.drop('insurance_premium', axis=1)
    y = df['insurance_premium']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = gb_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return gb_model, scaler

# Function to predict insurance premium for new data
def predict_premium(model, scaler, new_data):
    """
    Predict insurance premium for new data
    
    Parameters:
    new_data: DataFrame with the same features as training data
    """
    scaled_data = scaler.transform(new_data)
    prediction = model.predict(scaled_data)
    return prediction

if __name__ == "__main__":
    # Train the model
    model, scaler = create_insurance_model()
    
    # Example of using the model for prediction
    sample_data = pd.DataFrame({
        'age': [25],
        'vehicle_age': [5],
        'vehicle_value': [25000],
        'annual_mileage': [15000],
        'accident_history': [0]
    })
    
    predicted_premium = predict_premium(model, scaler, sample_data)
    print(f"\nPredicted Premium for sample data: ${predicted_premium[0]:.2f}") 