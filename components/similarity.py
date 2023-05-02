from collections import namedtuple

import tensorflow as tf
from IPython.core.display_functions import display


# def get_point_title_by_id(ID):
#     return list(merged_df[merged_df.ID == ID].name)[0]
#
# def get_point_id_by_name(name):
#     return list(merged_df[merged_df.name == name].ID)[0]


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


SimilarHouses = namedtuple('SimilarHouses', ['query_house_id', 'similar_house_ids', 'similarity_scores'])

def find_similar_houses(df, query_house_id, num_similar_houses):
    # Find the index of the query house in the dataframe
    query_house_index = df.index[df['ID'] == query_house_id].tolist()[0]

    # Extract the embeddings of all the houses in the dataframe
    embeddings = np.vstack(df['embeddings_healthcare'].values)

    # Compute the cosine similarity between the query house and all the other houses
    similarity_scores = cosine_similarity(embeddings[query_house_index].reshape(1, -1), embeddings)

    # Get the indices of the houses with the highest similarity scores
    similar_house_indices = np.argsort(similarity_scores)[0][::-1][:num_similar_houses + 1]

    # Get the house IDs and similarity scores of the similar houses
    similar_house_ids = df.loc[similar_house_indices, 'ID'].values
    similar_house_scores = similarity_scores[0, similar_house_indices]

    # Remove the query house from the list of similar houses
    query_house_pos = np.where(similar_house_ids == query_house_id)[0][0]
    similar_house_ids = np.delete(similar_house_ids, query_house_pos)
    similar_house_scores = np.delete(similar_house_scores, query_house_pos)

    # Create a dictionary of the query house ID and the IDs and similarity scores of the similar houses
    result = SimilarHouses(query_house_id, similar_house_ids.tolist(), similar_house_scores.tolist())

    return result


import folium


def create_map(similar_houses, df):
    # Get the coordinates of the query house
    query_house_coords = \
    df.loc[df['ID'] == similar_houses.query_house_id, ['y', 'x']].reset_index(drop=True).iloc[0].tolist()

    # Create a map centered on the query house
    map_center = query_house_coords
    m = folium.Map(location=map_center, zoom_start=10)

    # Add a marker for the query house
    folium.Marker(location=query_house_coords, popup=f"Query house {similar_houses.query_house_id}",
                  icon=folium.Icon(color='blue')).add_to(m)

    # Add markers for the similar houses
    for i, house_id in enumerate(similar_houses.similar_house_ids):
        coords = df.loc[df['ID'] == house_id, ['y', 'x']].reset_index(drop=True).iloc[0].tolist()
        score = round(similar_houses.similarity_scores[i], 2)
        folium.Marker(location=coords, popup=f"Similar house {house_id}\n, similarity score: {score}",
                      icon=folium.Icon(color='red')).add_to(m)

    # Display the map
    return m

"""

def plot_similarities(df, ):
    import folium

    def draw_marker(location, radius, color, map_obj):
        folium.CircleMarker(location=location, radius=radius, color=color).add_to(map_obj)

    def draw_pois(pois_df, map_obj):
        for _, row in pois_df.iterrows():
            y = row['y']
            x = row['x']
            degree = 4
            color = 'black'
            draw_marker((y, x), degree, color, map_obj)

    def draw_locations(locations_df, map_obj):
        for _, row in locations_df.iterrows():
            y = row['geometry'].y
            x = row['geometry'].x
            degree = row['degree']
            color = 'red'
            draw_marker((y, x), degree, color, map_obj)

    # Create map centered on first location in locations_df
    lat = geo_df1.geometry.y[0]
    lon = geo_df1.geometry.x[0]
    map_nl = folium.Map(location=(lat, lon), zoom_start=14)

    # Draw blue marker for center location
    draw_marker((lat, lon), 5, 'blue', map_nl)

    # Draw red markers for locations in locations_df
    draw_locations(geo_df1[1:], map_nl)

    # Draw black markers for POIs in pois_near_by
    draw_pois(pois_near_by, map_nl)

    # Display map
    display(map_nl)
    
"""