import concurrent.futures
import os

import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

import concurrent.futures
import numpy as np

from components.logs.Logger import get_logger
from enums.Category import Category

logger = get_logger(__name__)

from utils.cal_dist import calculate_dist


def create_edges(arr1, arr2, dist, graph=nx.Graph()):
    # Create edges in a graph based on a given list of data points

    if not arr1.any() or not arr2.any():
        logger.error("List1 or List2 is empty")
        return None

    min_dist = dist

    # Use a set to store the IDs of processed data points
    processed_ids = set()

    # Use a multiprocessing pool to parallelize the calculations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data_point_1 in tqdm(
                arr1, position=0, leave=True, desc="Creating the graph"):

            # Check if the data point has already been processed
            if data_point_1[0] in processed_ids:
                logger.info(f"Skipping already processed data point {data_point_1[0]}")
                continue

            # Add the data point's ID to the set of processed IDs
            processed_ids.add(data_point_1[0])

            # Calculate the distances between the data point and all other data points in list2
            distances = np.array(
                list(executor.map(calculate_dist, [(data_point_1[6], data_point_2[6]) for data_point_2 in arr2])))

            # make zero values max
            distances[distances == 0] = np.max(distances) + 1

            # Find the indices of the data points that are within the minimum distance
            indices = np.where(distances < min_dist)[0]

            if len(indices) == 0:
                indices = [np.argmin(distances)]
                index = indices[0]
                dp2 = arr2[index]
                dp2_id = dp2[0]
                logger.warning(f'Node is not connected. Connecting to closest node. \n'
                               f'Data point ID: {dp2_id}\n'
                               f'Distance: {distances[index]}')

            for index in indices:
                dp2 = arr2[index]
                dp2_id = dp2[0]
                weight = np.round(np.reciprocal(distances[index]), 1)
                graph.add_edge(data_point_1[0], dp2_id, weight=weight)

            # logger.info(f"Added edge between {data_point_1[0]} and {dp2_id} with weight {weight}")

    return graph


graphs_dir = "/home/astosi/PycharmProjects/graph_embeddings/data"


def get_poi_graph(province, category):
    directory = f"pois_graphs/{province}/{category.name}_{category.distance}km.edgelist"
    path = os.path.join(graphs_dir, directory)

    logger.info(f"Checking for existing POI graph at {path}...")

    try:
        pois_graph = nx.read_weighted_edgelist(path)
        logger.info(f"Loaded POI graph for {category.name} in {province} with {len(pois_graph)} nodes and"
                    f" {pois_graph.size()} edges.")
    except FileNotFoundError:
        logger.error(f"POI graph file not found: {category.name}_{category.distance}km.edgelist")
        pois_graph = None

    return pois_graph


def get_house_graph(house, pois_graph, pois_near_by, category: Category):
    house_id = house[0][0]
    directory = f"houses/graphs/{category.name}/{house_id}_{category.distance}km.edgelist"

    path = os.path.join(graphs_dir, directory)
    logger.info(f"Checking for already created graphs in {path}.")

    if os.path.exists(path):
        logger.info(f"Graph is found in {path}, for house {house_id}.")
        return nx.read_weighted_edgelist(path)

    new_graph = pois_graph.copy()

    logger.info("Creating graph for house")
    graph = create_edges(graph=new_graph, arr1=house, arr2=pois_near_by, dist=category.distance)

    nx.write_edgelist(graph, path, data=['weight'])
    logger.info(f"House graph is created and saved in {path}.")
    return graph


def get_edge_attr(graph):
    return nx.get_edge_attributes(graph, "weight")


def plot_graph(graph):
    plt.figure(figsize=(18, 18))

    G = graph
    graph_pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, graph_pos, node_size=12, node_color='red', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    # nx.draw_networkx_labels(G, graph_pos, font_size=2, font_family='sans-serif')

    plt.show()
    # plt.savefig("/content/drive/MyDrive/UH/utrecht_plot.png", dpi=500)
