import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    """
    Graph Neural Network for fraud detection.
    Uses two GCN layers followed by a linear classification layer.
    """
    
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of node features (default: 3 for one-hot encoded node types)
            hidden_dim: Dimension of hidden layer (default: 32)
            output_dim: Dimension of output (default: 2 for binary classification)
        """
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object with x (node features) and edge_index
            
        Returns:
            Logits for binary classification (fraud vs normal)
        """
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        # Linear classification layer
        return self.fc(x)
    
    def get_embeddings(self, data):
        """
        Get node embeddings from the second GCN layer.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings of dimension hidden_dim // 2
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
