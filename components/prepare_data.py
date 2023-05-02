import geopandas as gp
import numpy as np
import pandas as pd


from shapely import Point

from enums.Category import Category


def get_house_geopandas(house_id: int, address: str, province: str, x: float, y: float) -> gp.GeoDataFrame:
    # Turn coordinates into points
    point = Point(x, y)

    # To turn df to geopandas, generate a list of dictionaries
    house = [{'ID' : house_id , 'name': address, 'province': province, 'type': 'house', 'x': x, 'y': y}]

    # Using addresses and points we generate geodataframe
    houses_geo = gp.GeoDataFrame(house, geometry=[point], crs=4326)

    # Concat string to index
    # Rename the index column
    #houses_geo = houses_geo.reset_index().rename(columns={'index': 'ID'})
    #
    #houses_geo['ID'] = 'H' + str(house_id)

    return houses_geo


def get_house_geopandas_with_df(df: pd.DataFrame) -> gp.GeoDataFrame:
    # Get houses and eliminate which has same coords
    df1 = df.loc[:, ['address', 'province', 'x', 'y', 'price']].drop_duplicates(subset=['x', 'y'],
                                                                                keep='last').reset_index(drop=True)
    # We will use type property to distinguish from POI's
    df1['type'] = 'house'

    # Turn coordinates into points
    points_series = pd.Series(Point(x, y) for x, y in df1[['x', 'y']].values)

    # To turn df to geopandas, generate an array
    houses_array = [{'name': a.address, 'province': a.province, 'type': a.type, 'x': a.x, 'y': a.y} for _, a in
                    df1[['address', 'province', 'type', 'x', 'y']].iterrows()]

    # Using addresses and points we generate geodataframe
    houses_geo = gp.GeoDataFrame(houses_array, geometry=points_series, crs=4326)

    # Sort and reset index
    houses_geo = houses_geo.sort_values(by='x').reset_index()

    # Concat string to index
    houses_geo['index'] = 'H' + houses_geo.index.astype(str)

    # Rename the index column
    houses_geo = houses_geo.rename(columns={'index': 'ID'})

    return houses_geo


def get_pois(province: str, category: Category) -> gp.GeoDataFrame:
    # Read in the CSV file and drop duplicate rows based on the 'x' and 'y' columns
    pois_near_by = pd.read_csv(f"/home/astosi/UH/GraphThings/gsn/data/pois_by_province_2022/{province}/{category.name}.csv",
                               index_col=0).drop_duplicates(
        subset=['x', 'y'], keep='last')


    pois_near_by = pois_near_by[pois_near_by.type != 'veterinary']

    # Create a boolean mask indicating which rows have numeric values in the 'x' column
    numeric_mask = pois_near_by.x.apply(lambda x: isinstance(x, (int, np.int64, float, np.float64)))

    # Create a boolean mask indicating which rows have null values in the 'name' column
    null_mask = pois_near_by.name.isna()

    # Drop rows from the DataFrame where either the 'x' column is not numeric or the 'name' column is null
    pois_near_by = pois_near_by.drop(pois_near_by[~numeric_mask | null_mask].index).reset_index(drop=True)

    # Sort the DataFrame by the 'x' column
    pois_near_by = pois_near_by.sort_values(by='x')

    pois_near_by = pois_near_by.reset_index()
    # Add an 'ID' column to the DataFrame by concatenating the string "P" with the index values of each row
    pois_near_by['index'] = 'P' + pois_near_by.index.astype(str)

    pois_near_by = pois_near_by.rename(columns={'index': 'ID'})

    # Import the loads function from the shapely library
    from shapely import wkt

    # Convert the WKT strings in the 'geometry' column to shapely geometry objects
    pois_near_by['geometry'] = pois_near_by['geometry'].apply(wkt.loads)


    # Turn the object to geopandas
    pois_near_by = gp.GeoDataFrame(pois_near_by, crs=4326)

    pois_near_by.insert(loc=2, column='province', value=province)

    return pois_near_by
