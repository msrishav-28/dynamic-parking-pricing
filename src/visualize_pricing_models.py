import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the processed dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'processed_parking_data.csv')
try:
    df = pd.read_csv(data_path)
    print(f"Successfully loaded data from {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"Error: Could not find the processed data file at {data_path}")
    print("Please run the dynamic_pricing_model.py script first to generate the processed data.")
    exit(1)

# Create output directory path for saving plots
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

# Convert datetime column if it exists
if 'LastUpdatedDateTime' in df.columns:
    df['LastUpdatedDateTime'] = pd.to_datetime(df['LastUpdatedDateTime'])
else:
    print("Warning: LastUpdatedDateTime column not found. Time-based plots may not work correctly.")

# Set plot style
sns.set_style("whitegrid")

# 1. Compare Pricing Models (e.g., for a specific parking lot over time)
if 'LastUpdatedDateTime' in df.columns and 'SystemCodeNumber' in df.columns:
    # Select a parking lot for visualization (e.g., the first one in the dataset)
    sample_lot_code = df["SystemCodeNumber"].unique()[0]
    sample_lot_df = df[df["SystemCodeNumber"] == sample_lot_code].sort_values(by="LastUpdatedDateTime")

    plt.figure(figsize=(14, 7))
    plt.plot(sample_lot_df["LastUpdatedDateTime"], sample_lot_df["Price_Model1"], label="Model 1 (Baseline Linear)", linewidth=2)
    plt.plot(sample_lot_df["LastUpdatedDateTime"], sample_lot_df["Price_Model2"], label="Model 2 (Demand-Based)", linewidth=2)
    plt.plot(sample_lot_df["LastUpdatedDateTime"], sample_lot_df["Price_Model3"], label="Model 3 (Competitive)", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Parking Price Comparison for {sample_lot_code}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_comparison_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated price_comparison_over_time.png")
else:
    print("Skipping time series plot - required columns not found")

# 2. Impact of Occupancy Rate on Price (Model 2)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="OccupancyRate", y="Price_Model2", data=df, alpha=0.6)
plt.xlabel("Occupancy Rate")
plt.ylabel("Price (Model 2)")
plt.title("Impact of Occupancy Rate on Model 2 Price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "occupancy_impact_model2.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated occupancy_impact_model2.png")

# 3. Impact of Queue Length on Price (Model 2)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="QueueLength", y="Price_Model2", data=df, alpha=0.6)
plt.xlabel("Queue Length")
plt.ylabel("Price (Model 2)")
plt.title("Impact of Queue Length on Model 2 Price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "queue_impact_model2.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated queue_impact_model2.png")

# 4. Distribution of Prices for each model
plt.figure(figsize=(12, 6))
sns.histplot(df["Price_Model1"], color="skyblue", label="Model 1", kde=True, stat="density", linewidth=0, alpha=0.7)
sns.histplot(df["Price_Model2"], color="orange", label="Model 2", kde=True, stat="density", linewidth=0, alpha=0.7)
sns.histplot(df["Price_Model3"], color="green", label="Model 3", kde=True, stat="density", linewidth=0, alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Density")
plt.title("Distribution of Prices for Each Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated price_distribution.png")

# 5. Box plot of prices by VehicleType (Model 2)
plt.figure(figsize=(10, 6))
sns.boxplot(x="VehicleType", y="Price_Model2", data=df)
plt.xlabel("Vehicle Type")
plt.ylabel("Price (Model 2)")
plt.title("Price Distribution by Vehicle Type (Model 2)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_by_vehicletype_model2.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated price_by_vehicletype_model2.png")

# 6. Box plot of prices by TrafficConditionNearby (Model 2)
plt.figure(figsize=(10, 6))
sns.boxplot(x="TrafficConditionNearby", y="Price_Model2", data=df)
plt.xlabel("Traffic Condition Nearby")
plt.ylabel("Price (Model 2)")
plt.title("Price Distribution by Traffic Condition Nearby (Model 2)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_by_trafficcondition_model2.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated price_by_trafficcondition_model2.png")

# 7. Correlation Heatmap of numerical features and prices
plt.figure(figsize=(12, 8))
# Select only the columns that exist and are numerical
correlation_columns = []
potential_columns = ["OccupancyRate", "QueueLength", "IsSpecialDay", "VehicleTypeWeight", "TrafficConditionWeight", "Price_Model1", "Price_Model2", "Price_Model3"]

for col in potential_columns:
    if col in df.columns:
        correlation_columns.append(col)

correlation_matrix = df[correlation_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Matrix of Features and Prices")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Generated correlation_heatmap.png")

print(f"\nAll visualizations saved to: {output_dir}")
print("Summary of generated files:")
visualizations = [
    "price_comparison_over_time.png",
    "occupancy_impact_model2.png", 
    "queue_impact_model2.png",
    "price_distribution.png",
    "price_by_vehicletype_model2.png",
    "price_by_trafficcondition_model2.png",
    "correlation_heatmap.png"
]

for viz in visualizations:
    print(f"  - {viz}")

# Print some additional statistics
print("\n" + "="*50)
print("PRICING MODEL ANALYSIS SUMMARY")
print("="*50)

print(f"\nDataset Overview:")
print(f"  - Total records: {len(df):,}")
print(f"  - Unique parking lots: {df['SystemCodeNumber'].nunique()}")
print(f"  - Date range: {df['LastUpdatedDateTime'].min()} to {df['LastUpdatedDateTime'].max()}")

print(f"\nPricing Model Statistics:")
for model in ['Price_Model1', 'Price_Model2', 'Price_Model3']:
    if model in df.columns:
        mean_price = df[model].mean()
        std_price = df[model].std()
        min_price = df[model].min()
        max_price = df[model].max()
        print(f"  {model}:")
        print(f"    Mean: ${mean_price:.2f} | Std: ${std_price:.2f} | Range: ${min_price:.2f} - ${max_price:.2f}")

print(f"\nOccupancy Rate Analysis:")
print(f"  - Average occupancy rate: {df['OccupancyRate'].mean():.2%}")
print(f"  - Max occupancy rate: {df['OccupancyRate'].max():.2%}")

print(f"\nVehicle Type Distribution:")
vehicle_counts = df['VehicleType'].value_counts()
for vehicle_type, count in vehicle_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  - {vehicle_type}: {count:,} ({percentage:.1f}%)")

print(f"\nTraffic Condition Distribution:")
traffic_counts = df['TrafficConditionNearby'].value_counts()
for condition, count in traffic_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  - {condition}: {count:,} ({percentage:.1f}%)")

print("\n" + "="*50)
print("Analysis complete! Check the output directory for visualizations.")
print("="*50)