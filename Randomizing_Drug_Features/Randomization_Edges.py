import pandas as pd 
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import itertools
import numpy as np
import os
import random

for each in os.listdir("./Data/Drugs"):
    
    #from torch geometric to networkx
    edge_index = pd.read_csv(f"./Data/Drugs/{each}/Edge_Index_{each}.csv", sep='\t', index_col=0)
    MolGNet = pd.read_csv(f"./Data/Drugs/{each}/MolGNet_{each}.csv", sep='\t', index_col=0)

    edge_index_tensor = torch.tensor(edge_index.values)

    data = Data(edge_index=edge_index_tensor, num_nodes=MolGNet.shape[0])
    data_Graph = to_networkx(data, to_undirected=True) 

    #randomize the graph but keep the degrees for each node
    degrees = [data_Graph.degree[i] for i in range(len(data_Graph))]
    Graph_Randomized=nx.configuration_model(degrees)
    selfloops = list(nx.selfloop_edges(Graph_Randomized))
    Graph_Randomized = nx.Graph(Graph_Randomized)
    Graph_Randomized.remove_edges_from(selfloops)

    #from networkx to torch geometric
    data_Randomized = from_networkx(Graph_Randomized)

    #from torch geometric to csv with random new edges if missing
    GRAPH_df = pd.DataFrame(data_Randomized.edge_index)
    
    num_atoms = list(range(0, MolGNet.shape[0]))
    all_combinations = list(itertools.combinations(num_atoms,2))

    edge_index_combinations = []
    for i in range(GRAPH_df.shape[1]):
        edge_index_combinations.append((int(GRAPH_df.iloc[0,i]), int(GRAPH_df.iloc[1,i])))
        
    no_intersection = list(set(all_combinations) - set(edge_index_combinations))

    num_missing_edges=(int(edge_index.shape[1]/2)-int(GRAPH_df.shape[1]/2))
    for i in range(num_missing_edges):
        random_int = random.randint(0,len(no_intersection)-1)
        GRAPH_df[GRAPH_df.shape[1]]=[no_intersection[random_int][0],no_intersection[random_int][1]]
        GRAPH_df[GRAPH_df.shape[1]]=[no_intersection[random_int][1],no_intersection[random_int][0]]
        no_intersection.pop(random_int)
        
    GRAPH_df.to_csv(f"./Data/Drugs_Edges_Randomized/{each}/Edge_Index_{each}.csv", sep='\t')  
