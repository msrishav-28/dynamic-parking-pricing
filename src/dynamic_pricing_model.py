import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

# Load the dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset.csv')
df = pd.read_csv(data_path)

# Convert LastUpdatedDate and LastUpdatedTime to datetime objects
df["LastUpdatedDateTime"] = pd.to_datetime(df["LastUpdatedDate"] + " " + df["LastUpdatedTime"], format="%d-%m-%Y %H:%M:%S")

# Sort by SystemCodeNumber and LastUpdatedDateTime to ensure chronological order for each parking lot
df = df.sort_values(by=["SystemCodeNumber", "LastUpdatedDateTime"])

# Handle missing values and data quality issues
df = df.dropna(subset=['Capacity', 'Occupancy', 'Latitude', 'Longitude'])
df = df[df['Capacity'] > 0]  # Remove entries with zero or negative capacity

# Feature Engineering
df["OccupancyRate"] = np.clip(df["Occupancy"] / df["Capacity"], 0, 1)  # Clip to [0,1] range

vehicle_type_weights = {"car": 1.0, "bike": 0.5, "truck": 1.5, "cycle": 0.3}
df["VehicleTypeWeight"] = df["VehicleType"].map(vehicle_type_weights).fillna(1.0)  # Default weight for unknown types

traffic_condition_weights = {"low": 0.5, "average": 1.0, "high": 1.5}
df["TrafficConditionWeight"] = df["TrafficConditionNearby"].map(traffic_condition_weights).fillna(1.0)  # Default weight

# Model 1: Baseline Linear Model
# Pricet+1 = Pricet + α · (Occupancy / Capacity)
# Apply per parking space
def apply_baseline_linear_model(group, alpha=5.0):
    group = group.copy()
    group["Price_Model1"] = 10.0  # Base price
    for i in range(1, len(group)):
        prev_price = group.iloc[i-1]["Price_Model1"]
        occupancy_rate = group.iloc[i]["OccupancyRate"]
        new_price = prev_price + alpha * occupancy_rate
        # Ensure prices stay within reasonable bounds
        group.iloc[i, group.columns.get_loc("Price_Model1")] = max(5.0, min(50.0, new_price))
    return group

df = df.groupby("SystemCodeNumber", group_keys=False).apply(apply_baseline_linear_model).reset_index(drop=True)

# Model 2: Demand-Based Price Function
# Demand = α·(Occupancy/Capacity) + β·QueueLength − γ·Traffic + δ·IsSpecialDay + ε·VehicleTypeWeight
# Pricet = BasePrice · (1 + λ · NormalizedDemand)

def demand_based_model(df, alpha=1.0, beta=0.5, gamma=0.2, delta=2.0, epsilon=1.0, lambda_val=0.5):
    df = df.copy()
    df["Demand"] = (alpha * df["OccupancyRate"]) + \
                   (beta * df["QueueLength"]) - \
                   (gamma * df["TrafficConditionWeight"]) + \
                   (delta * df["IsSpecialDay"]) + \
                   (epsilon * df["VehicleTypeWeight"])

    # Normalize Demand (min-max normalization per parking lot)
    def safe_normalize(x):
        if x.max() == x.min():
            return pd.Series([0.5] * len(x), index=x.index)  # If all values are same, use 0.5
        return (x - x.min()) / (x.max() - x.min())
    
    df["NormalizedDemand"] = df.groupby("SystemCodeNumber")["Demand"].transform(safe_normalize)
    df["Price_Model2"] = 10.0 * (1 + lambda_val * df["NormalizedDemand"])
    
    return df

df = demand_based_model(df)

# Model 3: Competitive Pricing Model (Optimized)
# Calculate geographic proximity and factor in competitor prices

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Get unique parking lot locations and their SystemCodeNumbers
parking_lot_locations = df[["SystemCodeNumber", "Latitude", "Longitude"]].drop_duplicates().set_index("SystemCodeNumber")

# Pre-calculate distances between all parking lots
distances_matrix = pd.DataFrame(index=parking_lot_locations.index, columns=parking_lot_locations.index)
for idx1, loc1 in parking_lot_locations.iterrows():
    for idx2, loc2 in parking_lot_locations.iterrows():
        if idx1 != idx2:
            dist = haversine(loc1["Latitude"], loc1["Longitude"], loc2["Latitude"], loc2["Longitude"])
            distances_matrix.loc[idx1, idx2] = dist

# Convert to numeric, errors='coerce' will turn non-numeric into NaN
distances_matrix = distances_matrix.apply(pd.to_numeric, errors='coerce')
distances_matrix = distances_matrix.fillna(0)  # Fill diagonal with 0

# Calculate average competitor price for each parking lot
# This is done once per parking lot, not per row in the main DataFrame

# Create a dictionary to store average competitor prices for each SystemCodeNumber
avg_competitor_prices = {}
influence_radius = 5.0 # kilometers

for current_lot_code in parking_lot_locations.index:
    nearby_competitors = distances_matrix.loc[current_lot_code]
    nearby_competitors = nearby_competitors[(nearby_competitors > 0) & (nearby_competitors <= influence_radius)]
    
    if not nearby_competitors.empty:
        competitor_codes = nearby_competitors.index.tolist()
        # Get average Model 2 price of nearby competitors across all timestamps
        competitor_prices = df[df["SystemCodeNumber"].isin(competitor_codes)]["Price_Model2"]
        if not competitor_prices.empty:
            avg_comp_price = competitor_prices.mean()
            avg_competitor_prices[current_lot_code] = avg_comp_price
        else:
            avg_competitor_prices[current_lot_code] = 10.0  # Default base price
    else:
        avg_competitor_prices[current_lot_code] = 10.0  # Default base price if no competitors

# Map the pre-calculated average competitor prices back to the main DataFrame
df["AvgCompetitorPrice"] = df["SystemCodeNumber"].map(avg_competitor_prices)

# Now, apply Model 3 based on this average competitor price
price_sensitivity_model3 = 0.05  # Adjust this parameter

def calculate_competitive_price(row):
    if pd.isna(row["AvgCompetitorPrice"]) or row["AvgCompetitorPrice"] == 0:
        return row["Price_Model2"]
    
    price_diff_ratio = (row["Price_Model2"] - row["AvgCompetitorPrice"]) / max(row["Price_Model2"], 0.01)
    adjustment = price_sensitivity_model3 * price_diff_ratio
    return row["Price_Model2"] * (1 - adjustment)

df["Price_Model3"] = df.apply(calculate_competitive_price, axis=1)

# Ensure prices are within a reasonable range
df["Price_Model1"] = df["Price_Model1"].clip(5.0, 50.0)
df["Price_Model2"] = df["Price_Model2"].clip(5.0, 50.0)
df["Price_Model3"] = df["Price_Model3"].clip(5.0, 50.0)

# Save processed data to output directory
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'processed_parking_data.csv')
df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to {output_path}")

print("\nProcessed DataFrame Head with all features and Model 1, 2 & 3 prices:")
print(df[["SystemCodeNumber", "OccupancyRate", "Price_Model1", "Price_Model2", "Price_Model3", "Demand", "NormalizedDemand"]].head(10))

# Print summary statistics
print("\nSummary Statistics for Pricing Models:")
print(df[["Price_Model1", "Price_Model2", "Price_Model3"]].describe())

print("\nData Quality Check:")
print(f"Total records: {len(df)}")
print(f"Missing values in key columns:")
print(f"  - OccupancyRate: {df['OccupancyRate'].isna().sum()}")
print(f"  - Price_Model1: {df['Price_Model1'].isna().sum()}")
print(f"  - Price_Model2: {df['Price_Model2'].isna().sum()}")
print(f"  - Price_Model3: {df['Price_Model3'].isna().sum()}")
print(f"Unique parking lots: {df['SystemCodeNumber'].nunique()}")
print(f"Date range: {df['LastUpdatedDateTime'].min()} to {df['LastUpdatedDateTime'].max()}")