import mercantile, mapbox_vector_tile
import os
import requests
from shapely import geometry
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import math
import warnings
import osmnx as ox
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_city_boundary(city_name, country=None):
    if country:
        query = f"{city_name}, {country}"
    else:
        query = city_name
    
    try:
        boundary = ox.geocode_to_gdf(query)
        # print(boundary.head())
        return boundary
    except Exception as e:
        print(f"Error fetching boundary for {query}: {e}")
        return None

def generate_bounding_box(lon, lat, km=2.5):
    # 1 degree of latitude is approximately 111 km
    delta_lat = km / 111
    # 1 degree of longitude is approximately 111 km * cos(latitude) (adjusted by latitude)
    delta_lon = km / (111 * math.cos(math.radians(lat)))
    
    # Bounding box: [min_lon, min_lat, max_lon, max_lat]
    return [
        lon - delta_lon,  # min longitude
        lat - delta_lat,  # min latitude
        lon + delta_lon,  # max longitude
        lat + delta_lat   # max latitude
    ]

def get_tiles_from_bbox(bbox: list = None,
                        zoom = 17):
    return list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom))

def download_satellite_tiles_from_bbox(bbox, api_key: str, outpath):#, city = ''):
    # Make directory
    tile_path = outpath # + city)
    if not os.path.exists(tile_path):
            os.makedirs(tile_path)
    tiles = get_tiles_from_bbox(bbox)

    # Add spatial filter here

    for tile in tiles[:]:
        img_path = os.path.join(outpath + f'/{tile.z}_{tile.x}_{tile.y}.jpg')
        if os.path.exists(img_path):
            print(f'Skipping tile: {tile.z}/{tile.x}/{tile.y}, already exists.')
            continue
        tile_url = 'https://api.mapbox.com/v4/mapbox.satellite/{}/{}/{}@2x.jpg90?'.format(tile.z,tile.x,tile.y)
        response = requests.get(tile_url, params={'access_token': api_key})
        print(f'Downloaded tile: {tile.z}/{tile.x}/{tile.y}')

        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                
def get_tiles_gdf(tiles, bounds = None):
    x_list, y_list, z_list = [], [], []

    geom = []
    for tile in tiles:
        x_list.append(tile.x)
        y_list.append(tile.y)
        z_list.append(tile.z)
        geom.append(geometry.box(*mercantile.bounds(tile)))
    df = pd.DataFrame({'X': x_list, 'Y': y_list, 'Z': z_list})
    gdf = gpd.GeoDataFrame(data=df, crs = 'epsg:4326', geometry=geom)
    gdf['GID'] = range(len(gdf))

    if bounds is not None:
        intersect = sg.overlay(gdf)
        selected_grids = gdf[gdf['GID'].isin(list(intersect['GID']))]
        selected_grids = selected_grids[['X','Y','Z','geometry']]

        return selected_grids

    return gdf

def is_within_bbox(row, bbox):
    lon, lat = row['geometry'].centroid.x, row['geometry'].centroid.y
    # print(lon, lat)
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]

def get_grid_size(tile_gdf_proj):
    x_grid_size = (tile_gdf_proj.iloc[[0]].geometry.total_bounds[2] - tile_gdf_proj.iloc[[0]].geometry.total_bounds[0])/512
    y_grid_size = (tile_gdf_proj.iloc[[0]].geometry.total_bounds[3] - tile_gdf_proj.iloc[[0]].geometry.total_bounds[1])/512
    return x_grid_size, y_grid_size

def get_building_image_chips(building_proj, tiles_gdf_proj, add_context = False, pad_npixels = 64):
    # Project building to local coordinates
    building_chips = building_proj.copy()

    if add_context:
        geom_list = []
        x_grid_size, y_grid_size = get_grid_size(tiles_gdf_proj)
        building_chips['geometry'] = building_chips['geometry'].bounds.apply(lambda row: geometry.box(row['minx']-x_grid_size*pad_npixels,
                                                                                                      row['miny']-y_grid_size*pad_npixels,
                                                                                                      row['maxx']+x_grid_size*pad_npixels,
                                                                                                      row['maxy']+y_grid_size*pad_npixels), axis=1)
    else:
        # building_chips['geometry'] = building_chips['geometry'].bounds.apply(lambda row: geometry.box(row['minx'], row['miny'], row['maxx'], row['maxy']), axis=1)
        building_chips['geometry'] = building_chips['geometry'].bounds.apply(
            lambda row: geometry.box(
                row['minx'] - 10,  # expand 10 meters west
                row['miny'] - 10,  # expand 10 meters south
                row['maxx'] + 10,  # expand 10 meters east
                row['maxy'] + 10   # expand 10 meters north
            ),
            axis=1
        )
    return building_chips

def get_and_combine_tiles(tiles_gdf_proj, building_chips, data_folder, output_folder):

    # Create directory to store building satellite image chips
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    # Obtain chosen satellite images
    # Ensure output folder exists and collect existing building chips
    existing_chips = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_folder)
        if f.endswith('.png')
    }

    # Filter out building chips that already have data
    building_chips = building_chips[~building_chips['id'].isin(existing_chips)]
    if building_chips.empty:
        print("All building chips already exist. Skipping.")
        return

    # Perform spatial overlay to select relevant tiles
    chosen_tiles = gpd.overlay(tiles_gdf_proj, building_chips, how='intersection')

    x_grid_size, y_grid_size = get_grid_size(tiles_gdf_proj)

    problems = []
    uncovered = chosen_tiles['id'].unique()
    # for id in chosen_tiles['id'].unique():
    for id in tqdm(uncovered, desc="Processing building chips"):
        temp_gdf = chosen_tiles[chosen_tiles['id'] == id]
        columns = []
        # First loop over x values
        for x_val in temp_gdf['X'].unique():
            x_gdf = temp_gdf[temp_gdf['X'] == x_val]

            rasters = []
            for i, row in x_gdf.iterrows():
                raster_path = os.path.join(data_folder, f"{row['Z']}_{row['X']}_{row['Y']}.jpg")
                with rasterio.open(raster_path) as f:
                    raster = f.read()
                    rasters.append(raster)

            columns.append(np.concatenate((rasters), axis=1))

        try:
            # Combine satellite chips into single image
            img = np.concatenate((columns), axis=2)
        except (ValueError, SystemError) as e:
            print(f'Error in dim size for {id}. Error: {e}')
            problems.append(id)
            continue

        # Extract building from image
        b_minx, b_miny, b_maxx, b_maxy =  building_chips[building_chips['id'] == id]['geometry'].bounds.values[0]
        tile_minx, tile_miny, tile_maxx, tile_maxy = tiles_gdf_proj[tiles_gdf_proj['GID'].isin(list(chosen_tiles[chosen_tiles['id'] == id]['GID']))].total_bounds

        img_x_start = int((b_minx - tile_minx) / x_grid_size)
        img_y_start = int((tile_maxy-b_maxy) / y_grid_size)
        img_x_end = int((b_maxx - b_minx) / x_grid_size) + img_x_start
        img_y_end = img_y_start + int((b_maxy - b_miny) / y_grid_size)
        cropped_img = img[:, img_y_start:img_y_end, img_x_start:img_x_end]
        # print(f'Saving image chip for building: {id}.')
        # Save image chip for building
        try:
            im = Image.fromarray(np.transpose(cropped_img, (1, 2, 0)))
            im.save(os.path.join(os.getcwd(), output_folder, f"{id}.png"))
        except (ValueError, SystemError) as e:
            print(f'Cannot save: {id}. Error: {e}')
            problems.append(id)
            continue
        

def main():
    """Main function to download satellite data and generate building chips."""
    # Configuration
    CITY = 'Washington'
    OUTPUT_PATH = 'output'
    API_KEY = "YOUR_MAPBOX_API_KEY"  # Replace with your actual Mapbox API key
    ZOOM_LEVEL = 17
    
    # City-specific bounding box (consider moving to a config file)
    CITY_BBOX = {
        'Washington': [-77.12476352262264, 38.809679062449916, -76.91120400549497, 39.00473379901156]
    }
    
    bbox = CITY_BBOX[CITY]
    satellite_output_path = os.path.join(OUTPUT_PATH, f'satellite_data/{CITY}')
    chips_output_path = os.path.join(OUTPUT_PATH, f'sate_bd_no_context/{CITY}')
    
    # Generate tiles and filter by city boundary
    tiles = get_tiles_from_bbox(bbox, zoom=ZOOM_LEVEL)
    city_boundary_gdf = get_city_boundary(CITY)
    
    if city_boundary_gdf is not None:
        tiles_gdf = get_tiles_gdf(tiles)
        intersecting_tiles_gdf = gpd.overlay(tiles_gdf, city_boundary_gdf, how='intersection')
        intersecting_tile_ids = set(zip(intersecting_tiles_gdf['X'], intersecting_tiles_gdf['Y'], intersecting_tiles_gdf['Z']))
        tiles = [tile for tile in tiles if (tile.x, tile.y, tile.z) in intersecting_tile_ids]
        print(f'Filtered to {len(tiles)} tiles within city boundary')
    else:
        tiles_gdf = get_tiles_gdf(tiles)
        print(f'Using all {len(tiles)} tiles (no city boundary filter)')
    
    # Download satellite tiles
    download_satellite_tiles_from_bbox(bbox, API_KEY, outpath=satellite_output_path)
    
    # Process building footprints
    building_gdf = load_and_filter_buildings(CITY, bbox)
    
    # Generate building chips
    tiles_gdf = tiles_gdf.to_crs(epsg=3857)
    building_gdf = building_gdf.to_crs(epsg=3857)
    building_chips = get_building_image_chips(building_gdf, tiles_gdf, add_context=False)
    building_chips = building_chips.rename(columns={'building_id': 'id'})
    
    # Extract and save building image chips
    get_and_combine_tiles(tiles_gdf, building_chips, satellite_output_path, chips_output_path)

def load_and_filter_buildings(city, bbox):
    """Load building footprints and filter by bounding box."""
    gdf = gpd.read_file(f'footprint/{city}.geojson')
    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.drop_duplicates(subset='building_id')
    
    # Filter buildings within bounding box
    gdf['in_center'] = gdf.apply(lambda row: is_within_bbox(row, bbox), axis=1)
    gdf = gdf[gdf['in_center']].reset_index(drop=True).drop(columns=['in_center'])
    
    return gpd.GeoDataFrame(gdf, geometry='geometry', crs=gdf.crs)

if __name__ == "__main__":
    main()
    
    
    
