# Graph Embeddings

This README file provides an overview of the POI Embeddings Generator, a project that generates embeddings for houses based on their surrounding points of interest (POIs) using random walk and Skip-gram techniques.
## Requirements

    Python 3.6 or newer
    numpy
    pandas
    geopandas
    networkx
    TensorFlow 2.0 or newer

## Installation

Clone the repository:

```bash
git clone https://github.com/Astosu/graph-embeddings.
```


Navigate to the project directory:

```bash
cd poi-embeddings-generator
```
Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To use the embeddings generator, call the generate_embeddings function from embedding_generator.py with the appropriate parameters. This function will generate embeddings for the given house and return a pandas DataFrame containing the embeddings.

Parameters for the generate_embeddings function are as follows:

    house_id: A unique identifier for the house
    x: X-coordinate of the house's location
    y: Y-coordinate of the house's location
    address: Address of the house
    province: Province where the house is located
    category: An instance of the Category enumeration representing the POI category
    hparams: An instance of the Hyperparameters enumeration containing hyperparameter values for the random walk and Skip-gram models

Workflow

The workflow of the POI Embeddings Generator is as follows:

    Fetch the house's location and nearby POIs from the specified data sources.
    Generate a graph from the house's location and nearby POIs.
    Create a vocabulary for the graph and generate random walks on it.
    Prepare the dataset for training the Skip-gram model and train the model.
    Get the embedding for the house, save it, and store it in a pandas DataFrame.


## Theoretical Background

The POI Embeddings Generator project leverages concepts from graph theory, random walks, and natural language processing (NLP) techniques, specifically the Skip-gram model, to generate meaningful embeddings for houses based on their surrounding points of interest (POIs).
### Graph Theory

Graph theory is a branch of mathematics that deals with the study of graphs, which are mathematical structures used to model pairwise relations between objects. A graph is made up of vertices (also called nodes) and edges, which connect pairs of vertices. In this project, a graph is constructed with houses and their surrounding POIs as nodes, and the edges represent the spatial relationships between them.

Graph Theory
![image](https://miro.medium.com/v2/resize:fit:720/format:webp/1*9Ux80uESvaUdtkEyRMYajw.png)

Image Source: Towards Data Science
### Random Walks

Random walks are a sequence of steps on a graph, where each step is chosen randomly from the current node's neighbors. Random walks help in exploring the structure and properties of a graph. By performing random walks on the constructed graph, we can capture the relationships between houses and POIs in the form of sequences.

Random Walk

![image](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*3pmstIOig4Qc3lrQS4xrNg.png)

Image Source: Towards Data Science
Skip-gram Model

The Skip-gram model is a popular NLP technique used to generate word embeddings in a high-dimensional vector space. It is an unsupervised learning model that aims to predict the context words given a target word in a sliding window. In the POI Embeddings Generator project, the sequences generated from random walks are treated as sentences, where the houses and POIs are analogous to words in a sentence. By training a Skip-gram model on these sequences, we can generate embeddings for houses and POIs that capture their spatial relationships in the high-dimensional space.

### Skip-gram Model

![image](https://miro.medium.com/max/700/1*SR6l59udY05_bUICAjb6-w.png)

In summary, the process of generating embeddings for houses based on their surrounding POIs involves:

  Constructing a graph with houses and their surrounding POIs as nodes, and edges representing their spatial relationships.
  Performing random walks on the graph to capture relationships between houses and POIs in sequences.
  Training a Skip-gram model on the sequences generated from random walks to obtain embeddings for houses and POIs that capture their spatial relationships in a high-dimensional vector space.
