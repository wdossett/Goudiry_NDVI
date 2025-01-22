import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

long_results_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_samples_long.csv"
long_results = pd.read_csv(long_results_path)


#region linear trendlines by month
sns.lmplot(data=long_results, x="Dist_vill", y="NDVI", hue="Month", scatter = False)
plt.title("NDVI vs Distance by Month")
plt.show()
#endregion

#region non-linear trendlines by month
g = sns.lmplot(
    data=long_results,
    x="Dist_vill",
    y="NDVI",
    hue="Month",          # Group by month
    order=2,              # Polynomial regression of order 2
    scatter=False,        # Hide scatter points
    legend = False,
    height=6,
    aspect=1.5,
    line_kws={"linewidth": 2},
)

# Get the Axes object
ax = g.ax

# Loop through each line and add labels
for line, label in zip(ax.lines, long_results["Month"].unique()):
    # Get the line's data coordinates
    x = line.get_xdata()
    y = line.get_ydata()
    
    # Place the label near the end of the line
    ax.text(
        x[-1], y[-1],        # Coordinates to place the label
        str(label),          # Text for the label
        color=line.get_color(),
        fontsize=10,
        va="center",         # Vertical alignment
        ha="left",           # Horizontal alignment
    )

# Adjust labels and title
ax.set_title("Trendlines for NDVI by Month", fontsize=16)
ax.set_xlabel("Distance to Village (m)", fontsize=12)
ax.set_ylabel("NDVI", fontsize=12)
plt.tight_layout()
plt.show()
#endregion


#region non-linear trendlines by year

# Map month numbers to names
month_mapping = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 
    6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 
    11: "Nov", 12: "Dec"
}
long_results["MonthName"] = long_results["Month"].map(month_mapping)

# Sort data by month for proper x-axis order
long_results["MonthName"] = pd.Categorical(
    long_results["MonthName"], 
    categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ordered=True
)

# Polynomial fitting and plotting per year
plt.figure(figsize=(10, 6))

# Loop through unique years
for year in sorted(long_results["Year"].unique()):
    # Filter data for the current year
    year_data = long_results[long_results["Year"] == year]
    
    # Map months to numerical values for fitting
    x_vals = year_data["Month"]
    y_vals = year_data["NDVI"]
    
    # Fit a 3rd-degree polynomial
    coeffs = np.polyfit(x_vals, y_vals, deg=3)
    poly = np.poly1d(coeffs)
    
    # Generate smooth trendline values
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_smooth = poly(x_smooth)
    
    # Plot the trendline
    plt.plot(
        x_smooth, y_smooth, label=f"{year}", linewidth=2
    )

# Customize x-axis labels
plt.xticks(
    ticks=range(1, 13),  # Numerical values for months
    labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
plt.xlabel("Month", fontsize=12)
plt.ylabel("NDVI", fontsize=12)
plt.title("Polynomial Trendlines of NDVI Per Year", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()
#endregion

#calculate average slope, curve, and min NDVI for select months

select_months = [7, 10, 11]
select_results = long_results[long_results['Month'].isin(select_months)]
select_results['Dist_vill_km'] = select_results['Dist_vill']/1000

metrics = pd.DataFrame()

for month in select_months:
    month_results = select_results[select_results['Month'] == month].reset_index()
    month_name = month_results['MonthName'][0]

    #calculate coefficients
    coefficients = np.polyfit(month_results["Dist_vill_km"], month_results["NDVI"], 2)

    a, b, c, = coefficients

    # Distance range 
    x_min, x_max = 0, 17  

    # Average Slope
    average_slope = (a * (x_max**2 - x_min**2) + b * (x_max - x_min)) / (x_max - x_min)

    # Curvature Strength
    curvature_strength = abs(2 * a)

    # Minimum NDVI
    x_min_ndvi = -b / (2 * a)  # x-coordinate of minimum
    min_ndvi = a * x_min_ndvi**2 + b * x_min_ndvi + c
    
    month_dict = {'month':month_name, 'avg_slope':average_slope, 'curve':curvature_strength, 'ndvi_min':min_ndvi, 'ndvi_min_x':x_min_ndvi}
    month_df = pd.DataFrame([month_dict] )
    metrics = pd.concat([metrics, month_df], ignore_index= True)
