import os
import hickle as hkl
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, JumpingKnowledge
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Step 1: Custom Dataset for Drug Molecules
class DrugGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DrugGraphDataset, self).__init__(root, transform, pre_transform)
        self.file_list = [f for f in os.listdir(root) if f.endswith('.hkl')]

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # Load .hkl file
        file_path = os.path.join(self.root, self.file_list[idx])
        features, adj_list, _ = hkl.load(file_path)

        # Convert adjacency list to edge_index
        edge_index = []
        for src, neighbors in enumerate(adj_list):
            for dst in neighbors:
                edge_index.append([src, int(dst)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Convert node features
        x = torch.tensor(features, dtype=torch.float)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        data.name = self.file_list[idx].split('.')[0]  # Save PubChem ID as name
        return data

##############################第一版的药物GNN
# Step 2: Define GNN Model
class DrugGraphGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DrugGraphGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph Convolution Layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Global Pooling (graph-level feature extraction)
        x = global_mean_pool(x, batch)
        # Fully connected layer
        x = self.fc(x)
        return x

###############################第二版的药物GNN-GAT
class ImprovedDrugGraphGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(ImprovedDrugGraphGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        # for _ in range(num_layers - 2):
        #     self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.jump = JumpingKnowledge(mode='cat')  # Concatenate multi-layer features
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_list = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x, edge_index))
            x_list.append(x)
        x = self.jump(x_list)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


# Step 3: Train or Infer Graph Features
def process_and_extract_features(dataset_path, input_dim=75, hidden_dim=128, output_dim=24, batch_size=32):
    # Load dataset
    dataset = DrugGraphDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ImprovedDrugGraphGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.eval()  # For feature extraction, no training needed

    # Process all graphs and extract features
    drug_features = {}
    for batch in dataloader:
        with torch.no_grad():
            graph_features = model(batch)  # Graph-level features
            for idx, graph_feature in enumerate(graph_features):
                pubchem_id = batch.name[idx]
                drug_features[pubchem_id] = graph_feature.cpu().numpy()

    # Convert drug_features dictionary to DataFrame
    drug_feature_df = pd.DataFrame.from_dict(drug_features, orient='index')
    # Ensure index is sorted in ascending order
    drug_feature_df.index = drug_feature_df.index.astype(int)  # Convert index to integers
    drug_feature_df = drug_feature_df.sort_index()

    return drug_feature_df


# Step 4: Usage
if __name__ == "__main__":
    dataset_path = '../processed_data/24drug_graph_feat'  # Path to your .hkl files
    features = process_and_extract_features(dataset_path)
    print(f"Extracted features for {len(features)} drugs.")
    # # Convert drug_features dictionary to DataFrame
    # drug_feature_df = pd.DataFrame.from_dict(features, orient='index')
    # # Ensure index is sorted in ascending order
    # drug_feature_df.index = drug_feature_df.index.astype(int)  # Convert index to integers
    # drug_feature_df = drug_feature_df.sort_index()
    # drug_feature_df.index.name = 'PubChem_ID'
    #
    # Save to CSV
    features.to_csv("../processed_data/drug_graph_features.csv")