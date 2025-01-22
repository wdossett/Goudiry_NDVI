import os
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from rasterio.io import MemoryFile
import glob
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.fill import fillnodata
from scipy.spatial import cKDTree
from rasterstats import zonal_stats, raster_stats
import pandas as pd
from rasterio.features import geometry_mask

#region Clip Rasters

# Paths
raster_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\3rd data DL USGS EE"  # Folder containing rasters
vector_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_dept_GeoSen.shp"       # Path to the vector file
output_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Data DL USGS EE - clipped"       # Folder to save clipped rasters

# Load the vector polygon
vector_data = gpd.read_file(vector_path)
vector_geom = [mapping(vector_data.geometry[0])]  # Single geometry as a list

# Function to clip a raster
def clip_raster(raster_path, geometry, output_folder):
    with rasterio.open(raster_path) as src:
        # Clip the raster
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save clipped raster
        raster_name = os.path.basename(raster_path)
        output_path = os.path.join(output_folder, f"clipped_{raster_name}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

# Process all rasters in the folder

for raster_file in os.listdir(raster_folder):
    if raster_file.endswith('.TIF'):  # Filter for raster files
        raster_path = os.path.join(raster_folder, raster_file)
        clip_raster(raster_path, vector_geom, output_folder)

print("Clipping process completed for all rasters.")
#endregion

#region define NDVI, cloud cover functions

def calculate_ndvi(nir_path, red_path, output_id):
    """
    Calculate NDVI from near-infrared and red band raster layers.

    Parameters:
        nir_path (str): Path to the near-infrared (NIR) band raster.
        red_path (str): Path to the red band raster.
        output_id (str): Unique ID of NDVI output raster to be saved in memory.
    """
    
    #check if the output dictionary exists, create it au besoin
    try:
        ndvi_results
    except NameError:
        ndvi_results = {} 

    # Open NIR and Red bands
    with rasterio.open(nir_path) as nir:
        nir_band = nir.read(1).astype('float32')
        nir_meta = nir.meta.copy()

    with rasterio.open(red_path) as red:
        red_band = red.read(1).astype('float32')

    # Avoid division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.clip(ndvi, -1, 1)  # NDVI values should range between -1 and 1

    # Update metadata
    nir_meta.update({
        'dtype': 'float32',
        'count': 1
    })

    # Save the NDVI raster in memory
    output_data = {
        "data": ndvi,           # NDVI array
        "metadata": nir_meta    # Raster metadata
    }
    
    return output_data


def clip_cloud_cover(ndvi_ID, ndvi_dict, QA_folder):
    """
    Clip NDVI rasters to remove cloud cover, using the QA_pixel dataset from EarthExplorer.
    Assume the QA_pixel files and NDVI files have the same identifying information.

    Args:
        ndvi_ID (str): Unique ID of NDVI output raster saved in memory
        ndvi_dict (dict): dictionary of data on the NDVI raster
        QA_folder (str): Path to folder with the QA_pixel rasters
    """

    # Path to the corresponding QA_PIXEL raster
    qa_pixel_path_str = 'clipped_' + ndvi_ID + '_QA_PIXEL.TIF'
    qa_pixel_path = os.path.join(QA_folder, qa_pixel_path_str) 
    
    # Check if the QA_PIXEL raster exists
    if not os.path.exists(qa_pixel_path):
        raise FileNotFoundError(f"QA_PIXEL file not found for NDVI ID: {ndvi_ID}")

    ndvi_data = ndvi_dict['data'] 
    ndvi_meta = ndvi_dict['metadata']  
    
    # Open the QA_PIXEL raster
    with rasterio.open(qa_pixel_path) as qa_src:
        # Read QA_PIXEL data
        qa_data = qa_src.read(1).astype("int32")
        
        # Create a mask where QA_PIXEL equals 21824
        valid_mask = (qa_data == 21824)
        
        # Apply the mask to the NDVI data
        ndvi_clipped = np.where(valid_mask, ndvi_data, np.nan)  # Replace non-matching pixels with NaN
        
        # Update metadata for the clipped raster
        clipped_meta = ndvi_meta.copy()
        clipped_meta.update(dtype="float32", nodata=np.nan)

    # Save the clipped NDVI raster in memory
    
    clipped_data =  {"data": ndvi_clipped, "metadata": clipped_meta}
    
    return clipped_data 

#endregion


#region Calculate NDVI


# Define the folders containing your files
folder_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Data DL USGS EE - clipped"
QA_folder_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Cloud QA clipped"

# Find all files with 'B4' and 'B5' in their names - red and NIR bands
red_files_raw = glob.glob(os.path.join(folder_path, "*B4*"))
nir_files_raw = glob.glob(os.path.join(folder_path, "*B5*"))

#create function to extract identifying information
def preprocess_filename(filename):
    return filename.replace("clipped_", "").replace("_SR_B5.TIF", "").replace("_SR_B4.TIF", "").replace("_QA_PIXEL.TIF","")

# Process the red and NIR file lists into dictionaries
red_files = {preprocess_filename(os.path.basename(f)): f for f in red_files_raw if f.endswith('.TIF')}
nir_files = {preprocess_filename(os.path.basename(f)): f for f in nir_files_raw if f.endswith('.TIF')}

#process the QA folder into dictionary
QA_files = {preprocess_filename(f): os.path.join(QA_folder_path, f) for f in os.listdir(QA_folder_path) if f.endswith('.TIF')}


# Create a dictionary to store NDVI rasters in memory
ndvi_results = {}

# count errors and successes
NDVI_counter = 0
missing_band_counter = 0 
missing_QA_counter = 0 

QA_missing = []
band4_missing = []

#process all the rasters
for file_name in nir_files: 
    
    if file_name in red_files:  #check if we have band4
        
        if file_name in QA_files: #check if we have the QA raster
            nir_path = nir_files[file_name]
            red_path = red_files[file_name]
            ndvi_dict = calculate_ndvi(nir_path, red_path, file_name) #calculate raw NDVI 
            
            ndvi_cloudless = clip_cloud_cover(file_name, ndvi_dict, QA_folder_path) #clip NDVI raster by cloud cover
            
            ndvi_results[file_name] = ndvi_cloudless #add to dictionary of results
            NDVI_counter += 1
        else: 
            #print(f"No matching file for {file_name} in QA folder.")
            missing_QA_counter +=1
            QA_missing.append(file_name)
    else:
        #print(f"No matching file for {file_name} in Band 4 folder.")
        missing_band_counter +=1
        band4_missing.append(file_name)

#update user 
print(f"{NDVI_counter} NDVI rasters saved.")
print(f"{missing_band_counter} missing band 4s.")
print(f"{missing_QA_counter} missing QA rasters.")

'''#save to disc, if necessary

output_folder = "C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI_clipped"

# Ensure the folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save each item in the dictionary as a separate GeoTIFF
for key, value in ndvi_results.items():
    output_path = os.path.join(output_folder, f"{key}NDVI_clipped.tif")
    
    # Save the data as a GeoTIFF
    with rasterio.open(output_path, 'w', driver=value['metadata']['driver'],
                       count=value['metadata']['count'], dtype=value['metadata']['dtype'],
                       crs=value['metadata']['crs'], transform=value['metadata']['transform'],
                       width=value['metadata']['width'], height=value['metadata']['height'],
                       nodata=value['metadata']['nodata']) as dst:
        dst.write(value['data'], 1)

'''

#endregion

#align raster extents
#region create an empty raster w/ the extent of the clip vector
#get vector to create empty raster 
vector = gpd.read_file(vector_path)
bounds = vector.total_bounds  # Get extent [minx, miny, maxx, maxy]

#get pixel size from last NDVI raster calculated
transform = ndvi_cloudless['metadata']['transform']

pixel_width = transform.a  # Horizontal pixel size (x-resolution)
pixel_height = -transform.e  # Vertical pixel size (y-resolution, negate for positive value)

#resolution tuple from pixel size
resolution = (pixel_width, pixel_height)

# Define raster dimensions and transform
minx, miny, maxx, maxy = bounds
width = int((maxx - minx) / resolution[0]) 
height = int((maxy - miny) / resolution[1])
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Define metadata
meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': np.nan,
    'width': width,
    'height': height,
    'count': 1,
    'crs': vector.crs,
    'transform': transform
}

#create empty raster in a file location
empty_data = np.full((height, width), np.nan, dtype='float32')

with rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_empty.TIF", "w", **meta) as dest:
    dest.write(np.where(np.isnan(empty_data), meta["nodata"], empty_data), 1)
empty_raster_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_empty.TIF"

#endregion

#region stack empty and NDVI rasters
#func to stack NDVI rasters on top of the empty raster so that each NDVI raster has the same extent
def add_raster_to_empty(empty_raster_path, input_raster_path, output_path, raster_id):
    """
    Overlay a raster onto an empty raster with matching extent and resolution.
    
    Args:
        empty_raster_path (str): path to the empty raster file.
        input_raster_path (str): path to the raster file to be added.
        output_path (str): Path to save the resulting raster.
    """

    with rasterio.open(empty_raster_path) as empty_raster, rasterio.open(input_raster_path) as input_raster:
        #set up metadata for aligned raster
        aligned_meta = empty_raster.meta.copy()

        # Create destination array
        aligned_array = np.empty((empty_raster.height, empty_raster.width), dtype=empty_raster.dtypes[0])

        # Reproject input raster to match empty raster
        reproject(
            source=rasterio.band(input_raster, 1),
            destination=aligned_array,
            src_transform=input_raster.transform,
            src_crs=input_raster.crs,
            dst_transform=empty_raster.transform,
            dst_crs=empty_raster.crs,
            resampling=Resampling.nearest,
        )

    with rasterio.open(f'{output_path}\{raster_id}_cloudlessNDVI.TIF', "w", **aligned_meta) as dest:
        dest.write(aligned_array, 1)
        #dest.write(np.where(np.isnan(mean_stack), output_meta["nodata"], mean_stack), 1)
    
  
#set variables
input_raster_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_result_forprocessing.TIF"
output_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI wo clouds"
counter = 0

#testing_subset = {key: value for key, value in ndvi_results.items() if '202403' in key }
#last_8_items = dict(list(ndvi_results.items())[-5:])

#iterate through the NDVI rasters to make them each the right extent
for result, data in ndvi_results.items():
    #write the NDVI raster to a raster file so we can use rasterio on it
    meta = data['metadata']
    ndvi_processing_data = data['data']
    
    with rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_result_forprocessing.TIF", "w", **meta) as dest:
        dest.write(np.where(np.isnan(ndvi_processing_data), meta["nodata"], ndvi_processing_data), 1)
    
    #run the function
    add_raster_to_empty(empty_raster_path, input_raster_path, output_path, result)
    
    counter += 1

print(f'{counter} NDVI raster extents modified')

#endregion


#region avg by month

#define func to average all the NDVI rasters in the same month
def get_monthly_ndvi(input_folder_path, output_folder):
    """
    Calculate monthly average NDVI from daily NDVI rasters.

    Parameters:
        input_folder_path (str): Path to folder containing NDVI raster files.
        output_folder (str): Path to save the monthly average NDVI rasters.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    #list raster files in input folder
    raster_files = [f for f in os.listdir(input_folder_path) if f.endswith('.TIF')]

    # Group rasters by month
    rasters_by_month = defaultdict(list) 
    for file in raster_files:
        # Assumes filenames like "LC08_L2SP_202050_20240706_20240712_02_T1_cloudlessNDVI.TIF"
        month = file.split('_')[3][:6]  # Extract YYYYMM

        rasters_by_month[month].append(file)

    # Process each month
    for month, files in rasters_by_month.items():
        
        rasters = []
        bounds = []
        
        for file in files:
            raster_path = os.path.join(input_folder_path,  file)
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                nodata_value = src.nodata if src.nodata else -9999
                data = np.where(data == nodata_value, np.nan, data)  # Mask null values
                rasters.append(data)
                bounds_unique = src.bounds
                bounds.append(bounds_unique)

        # Stack and compute the mean ignoring NaN values
        stacked_rasters = np.array(rasters)
        monthly_mean = np.nanmean(stacked_rasters, axis=0)

        # Save the averaged raster for the month
        output_filename = os.path.join(output_folder, f"NDVI_avg_{month}.TIF")
        with rasterio.open(raster_path) as src:
            output_meta = src.meta.copy()
            output_meta.update({
                "driver": "GTiff",
                "height": monthly_mean.shape[0],
                "width": monthly_mean.shape[1],
                "nodata": -9999
            })
        
        # Save the averaged raster
        with rasterio.open(output_filename, 'w', **output_meta) as dst:
            dst.write(np.where(np.isnan(monthly_mean), -9999, monthly_mean), 1)

        print(f"{month} processed")

# Input and output paths
input_folder_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI wo clouds\Problem children 2"
output_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI monthly averages"

# Run the processing
get_monthly_ndvi(input_folder_path, output_folder)

#endregion

#region clip to remove crop and built-up area

#vectorize raster with crop and built-up area set to 0 - use that as the vector mask

#Load the vector layer and filter 
land_vector_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\terrascope land use\land_use_vector.shp"
gdf = gpd.read_file(land_vector_path)
filtered_gdf = gdf[gdf['DN'] == 1]  # Filter polygons where DN = 1

#Extract geometry from the filtered GeoDataFrame
geometries = filtered_gdf.geometry


#iterate through each raster
raster_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI monthly averages"
output_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI monthly averages no cropland"

for ndvi_path in os.listdir(raster_folder):
    if ndvi_path.endswith(".TIF"):
        raster_path = os.path.join(raster_folder, ndvi_path)
        output_raster_path = os.path.join(output_folder, f"clipped_{ndvi_path}")
        
        # Ensure the CRS matches between vector and raster
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if filtered_gdf.crs != raster_crs:
                filtered_gdf = filtered_gdf.to_crs(raster_crs)

        #Clip the raster using the geometries as mask
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()

        # update metadata for the output raster
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Write the clipped raster to a new file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Clipped {ndvi_path} raster saved.")
#endregion

# distance to nearest village

#region create distance raster
# Load the vector file (villages)
villages = gpd.read_file(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_villages.shp")

# Load the blank raster file to use as template
raster = rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_empty.TIF")

# Reproject villages to match raster CRS
villages = villages.to_crs(raster.crs)

# Extract village coordinates
village_coords = [(x, y) for x, y in zip(villages.geometry.x, villages.geometry.y)]

# Create a grid of pixel center coordinates
rows, cols = np.indices((raster.height, raster.width))
x, y = raster.xy(rows.flatten(), cols.flatten())
pixel_coords = np.column_stack([x, y])

# Use KDTree for fast nearest-neighbor search
tree = cKDTree(village_coords)
distances, _ = tree.query(pixel_coords)

# Reshape the distances to match the raster shape
distance_raster = distances.reshape(raster.height, raster.width)

# Save the distance raster
output_file = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\distance_raster.TIF"
transform = raster.transform

with rasterio.open(
    output_file,
    'w',
    driver='GTiff',
    height=distance_raster.shape[0],
    width=distance_raster.shape[1],
    count=1,
    dtype=distance_raster.dtype,
    crs=raster.crs,
    transform=transform,
) as dst:
    dst.write(distance_raster, 1)
#endregion

#create sample points in QGIS, give each a distance value

#create a database of distance values and NDVI values by year and month for analysis

# Path to the folder containing NDVI rasters
raster_files = glob.glob(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\NDVI monthly averages no cropland\*.TIF")

# Path to the sample points shapefile
points_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_sample_points.shp"

# Load the points as a GeoDataFrame
points_gdf = gpd.read_file(points_path)

# Create an empty DataFrame to store results
results = points_gdf.copy()
# Drop unwanted columns
results = results.drop(columns=["left", "right", "top", "bottom", "row_index", "col_index", 'geometry'])

# Sample each raster
for raster_path in raster_files:
    # Extract the year and month from filename
    yearmonth = raster_path.split("/")[-1].split("_")[3].split(".")[0]
    year = yearmonth[0:4]
    month = yearmonth[4:]
    year_month = year + "_" + month    

    # Open the raster
    with rasterio.open(raster_path) as src:
        # Transform points to the raster's CRS
        points_gdf = points_gdf.to_crs(src.crs)

        # Extract raster values for each point
        coords = [(x, y) for x, y in zip(points_gdf.geometry.x, points_gdf.geometry.y)]
        sampled_values = list(src.sample(coords))

        # Add the sampled values as a new column
        results[year_month] = [value[0] for value in sampled_values]


# Reshape the results DataFrame for easier analysis
# Convert wide format (Year_Month as columns) to long format
long_results = pd.melt(
    frame=results, 
    id_vars=["id", "Dist_vill", 'geometry'],
    var_name="Year_Month",
    value_name="NDVI")

# Split Year_Month into separate columns
long_results[["Year", "Month"]] = long_results["Year_Month"].str.split("_", expand=True)
long_results.drop(columns=["Year_Month"], inplace=True)
#remove null values
long_results = long_results[long_results["NDVI"] != -9999]
long_results.reset_index(drop=True, inplace=True)

# Save the long-format results to a CSV file
long_results.to_csv(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_samples_long.csv", index=False)

#filter out every other point to reduce size
filtered_results = long_results[long_results["id"] % 2 == 0]
filtered_results.to_csv(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_samples_long_filtered.csv", index=False)
