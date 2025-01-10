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
from rasterio.fill import fillnodata

#region Clip Rasters

# Paths
raster_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\2nd Data DL USGS EE"  # Folder containing rasters
vector_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_dept_GeoSen.shp"       # Path to the vector file
output_folder = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\1st Data DL USGS EE - clipped"       # Folder to save clipped rasters

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
folder_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\1st data DL USGS EE - clipped"
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


#region align extent of NDVI rasters

#region start by creating an empty raster w/ the extent of the clip vector
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

#create empty raster

'''#in memory

empty_raster = {
    'data': np.full((height, width), np.nan, dtype='float32'), 
    'metadata': meta
}'''

#in a file location
empty_data = np.full((height, width), np.nan, dtype='float32')

with rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_empty.TIF", "w", **meta) as dest:
    dest.write(np.where(np.isnan(empty_data), meta["nodata"], empty_data), 1)

#endregion

#EXAMPLE
meta = ndvi_cloudless['metadata']
ndvi_test_data = ndvi_cloudless['data']
with rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_test.TIF", "w", **meta) as dest:
    dest.write(np.where(np.isnan(ndvi_test_data), meta["nodata"], ndvi_test_data), 1)

empty_raster_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\Goudiry_empty.TIF"
input_raster_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_test.TIF"
output_path = r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\test_output.TIF"

#func to stack NDVI rasters on top of the empty raster so that each NDVI raster has the same extent
def add_raster_to_empty(empty_raster_path, input_raster_path, output_path):
    """
    Overlay a raster onto an empty raster with matching extent and resolution.
    
    Args:
        empty_raster_path (str): path to the empty raster file.
        input_raster_path (str): path to the raster file to be added.
        output_path (str): Path to save the resulting raster.
    """

    with rasterio.open(empty_raster_path) as empty_raster, rasterio.open(input_raster_path) as input_raster:
        # Ensure both rasters have the same shape and resolution by merging
        merged_array, merged_transform = merge([empty_raster, input_raster], res=empty_raster.res, method="first")

        # Stack the merged rasters and handle nodata values
        nodata_value = empty_raster.nodata if empty_raster.nodata else -9999
        stack = np.where(merged_array == nodata_value, np.nan, merged_array)

        # Compute the mean ignoring NaN
        mean_stack = np.nanmean(stack, axis=0)
    
        # Save the resulting raster
        output_meta = empty_raster.meta.copy()
        output_meta.update({
            "driver": "GTiff",
            "height": mean_stack.shape[0],
            "width": mean_stack.shape[1],
            "transform": merged_transform,
            "nodata": -9999
        })

        with rasterio.open(output_path, "w", **output_meta) as dest:
            dest.write(np.where(np.isnan(mean_stack), output_meta["nodata"], mean_stack), 1)
    
    
    
    ''' in memory
    # Extract data and metadata from the dictionaries
    empty_data = empty_raster['data']
    empty_meta = empty_raster['metadata']

    ndvi_data = input_raster['data']
    ndvi_meta = input_raster['metadata']
    

    # Create an aligned NDVI raster to match the empty raster
    aligned_ndvi = np.full_like(empty_data, np.nan, dtype='float32')  # Initialize with NaN

    # Reproject NDVI to match the empty raster's extent and transform
    reproject(
        source=ndvi_data,
        destination=aligned_ndvi,
        src_transform=ndvi_meta['transform'],
        src_crs=ndvi_meta['crs'],
        dst_transform=empty_meta['transform'],
        dst_crs=empty_meta['crs'],
        resampling=Resampling.nearest  # Choose resampling method
    )

    # Stack the empty raster and the aligned NDVI raster
    stacked_rasters = np.stack([empty_data, aligned_ndvi], axis=0)

    print("Stacked raster shape:", stacked_rasters.shape)
        

    with MemoryFile() as memfile:
        with memfile.open(**empty_raster['metadata']) as dst: #store the empty raster in memory
            dst.write(empty_raster['data'], 1) #write empty raster data
            
        # Use the memory file
        with memfile.open() as src:
            data = src.read(1)  # Read the first band
            print(data)
        
        # Calculate window of overlap
        window = rasterio.windows.from_bounds(
            *src.bounds, transform=empty_src.transform, width=empty_meta['width'], height=empty_meta['height']
        )
        
        # Insert input data into the empty raster
        window = window.round_offsets()
        empty_data[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width] = input_data

        # Save the combined raster
        with rasterio.open(output_path, 'w', **empty_meta) as dst:
            dst.write(empty_data, 1)
    '''

#define empty raster pat
for result, data in ndvi_results.items():
    #write the result to a raster so we can use rasterio on it
    meta = data['metadata']
    ndvi_processing_data = data['data']
    
    with rasterio.open(r"C:\Users\user\Documents\Padova\GIS Applications\Lab Project\ndvi_result_forprocessing.TIF", "w", **meta) as dest:
        dest.write(np.where(np.isnan(ndvi_processing_data), meta["nodata"], ndvi_processing_data), 1)
    
    
    
# Example usage
vector_path = "path/to/vector.shp"
empty_raster_path = "path/to/empty_raster.tif"
input_raster_path = "path/to/raster.tif"
output_path = "path/to/output_raster.tif"



# Overlay a raster onto the empty raster
add_raster_to_empty(empty_raster_path, input_raster_path, output_path)

#endregion

#region avg by month

def get_monthly_ndvi(input_dict, output_folder):
    """
    Calculate monthly average NDVI from daily NDVI rasters.

    Parameters:
        input_dict (dict): Dictionary of dictionary containing NDVI rasters.
        output_folder (str): Path to save the monthly average NDVI rasters.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Group rasters by month
    rasters_by_month = defaultdict(list) 
    for id, result in input_dict.items():
        # Example assumes filenames like "ndvi_LC08_L2SP_202050_20240706_20240712_02_T1_SR"
        month = id.split('_')[4][:6]  # Extract YYYYMM
        rasters_by_month[month].append(result)

    # Process each month
    for month, raster_ids in rasters_by_month.items():
        print(f"Processing month: {month}")
        
        # Read and stack NDVI rasters for the month
        ndvi_stack = []
        meta = None
        
        # Iterate over key-value pairs in rasters_by_month
        for month, ndvi_data_dict_list in rasters_by_month.items():
            for ndvi_data_dict in ndvi_data_dict_list:  # Now iterating through the list of dictionaries for each month
                ndvi_stack.append(ndvi_data_dict['data'].astype('float32'))
                
                # Set metadata for the first NDVI data dictionary
                if meta is None:
                    meta = ndvi_data_dict['metadata']

        # Convert to 3D NumPy array (time, rows, cols)
        ndvi_stack = np.stack(ndvi_stack)
        
        #MAKE ALL THE STACKS HAVE THE SAME SHAPE, OR GET AROUND IT 

        # Compute pixel-wise mean
        monthly_mean = np.mean(ndvi_stack, axis=0)

        # Update metadata for the output raster
        meta.update(dtype='float32', count=1)

        # Save the monthly average NDVI raster
        output_path = os.path.join(output_folder, f"monthly_avg_ndvi_{month}.tif")
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(monthly_mean, 1)

        print(f"Saved monthly average NDVI to {output_path}")

# Input and output paths
input_folder = 'path/to/ndvi_rasters'
output_folder = 'path/to/output_folder'

# Run the processing
get_monthly_ndvi(input_folder, output_folder)

#endregion

