import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def build_graph(csv_path):
    """
    Build a NetworkX graph from transaction data.
    
    Args:
        csv_path: Path to CSV file with columns: user_id, account_id, transaction_id, amount, fraud
        
    Returns:
        NetworkX graph with user, account, and transaction nodes
    """
    df = pd.read_csv(csv_path)
    G = nx.Graph()
    
    # Keep track of added nodes for consistent attribute assignment
    added_nodes = set()
    
    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        account = f"account_{row['account_id']}"
        txn = f"txn_{row['transaction_id']}"
        
        # Ensure all nodes have 'node_type' and 'label' attributes
        if user not in added_nodes:
            G.add_node(user, node_type=0, label=0)  # User node
            added_nodes.add(user)
        
        if account not in added_nodes:
            G.add_node(account, node_type=1, label=0)  # Account node
            added_nodes.add(account)
        
        if txn not in added_nodes:
            G.add_node(txn, node_type=2, label=row['fraud'])  # Transaction node with fraud label
            added_nodes.add(txn)
        
        # Connect user to account and account to transaction
        G.add_edge(user, account)
        G.add_edge(account, txn)
    
    return G

def to_pyg_data(G):
    """
    Convert NetworkX graph to PyTorch Geometric Data format.
    
    Args:
        G: NetworkX graph
        
    Returns:
        PyTorch Geometric Data object ready for GNN
    """
    data = from_networkx(G)
    
    # Extract node types and create one-hot features
    node_types = [G.nodes[n]['node_type'] for n in G.nodes]
    data.x = torch.nn.functional.one_hot(
        torch.tensor(node_types),
        num_classes=3
    ).float()
    
    # Create labels and masks
    labels, mask = [], []
    for n in G.nodes:
        if G.nodes[n]['node_type'] == 2:  # Transaction nodes
            labels.append(G.nodes[n]['label'])
            mask.append(True)
        else:
            labels.append(0)  # Placeholder for non-transaction nodes
            mask.append(False)
    
    data.y = torch.tensor(labels)
    data.txn_mask = torch.tensor(mask)
    
    return data
