# Graph Neural Network for Fraud Detection

## Overview
This project implements a Graph Neural Network (GNN) to detect fraudulent transactions by modeling users, accounts, and transactions as a graph. The system uses a Graph Convolutional Network (GCN) architecture built with PyTorch Geometric to identify patterns indicative of fraudulent activity.

## Technologies Used
- **PyTorch Geometric**: For graph neural network implementation
- **PyTorch**: Deep learning framework
- **NetworkX**: Graph preprocessing and manipulation
- **Python**: Primary programming language
- **Jupyter/Colab**: For experimentation and visualization
- **scikit-learn**: For evaluation metrics
- **pandas**: Data manipulation
- **matplotlib**: Visualization

## Project Structure

### Key Requirements
1. **Graph Construction**: Create transaction graphs with nodes for users, accounts, and transactions
2. **GNN Architecture**: Implement GCN or GAT architecture in PyTorch Geometric
3. **Training Pipeline**: Train on financial transaction dataset with known fraud labels
4. **Embeddings**: Generate node embeddings and classify fraudulent transactions
5. **Evaluation**: Compute precision, recall, and AUC metrics

## Data Structure

The project creates a heterogeneous graph with three node types:
- **User Nodes** (node_type=0): Represent individual users
- **Account Nodes** (node_type=1): Represent financial accounts
- **Transaction Nodes** (node_type=2): Represent individual transactions with fraud labels

Edges connect users to accounts and accounts to transactions, capturing the relationships within the financial ecosystem.

## Model Architecture

### GNN Model
```
Input Features: One-hot encoded node types (3 features)
  ↓
GCN Layer 1: 3 → 32 dimensions
  ↓
ReLU Activation
  ↓
GCN Layer 2: 32 → 16 dimensions
  ↓
Linear Layer: 16 → 2 classes (Normal/Fraud)
  ↓
Output: Class probabilities
```

## Key Components

### 1. Graph Construction (`build_graph`)
- Reads transaction data from CSV
- Creates NetworkX graph with user, account, and transaction nodes
- Establishes edges between users↔accounts↔transactions
- Assigns node attributes for type identification

### 2. Data Conversion (`to_pyg_data`)
- Converts NetworkX graph to PyTorch Geometric Data format
- Generates one-hot encoded node features
- Creates labels and masks for training
- Separates transaction nodes for fraud classification

### 3. Training Pipeline
- Optimizer: Adam with learning rate 0.01
- Loss Function: Cross-Entropy Loss
- Epochs: 50
- Only trains on transaction nodes (where fraud labels exist)

### 4. Evaluation Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC (Area Under ROC Curve)**: Measures discrimination ability
- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curve**: Visualizes true positive rate vs false positive rate

## Results

Example performance on sample dataset:
- **Precision**: 0.500
- **Recall**: 1.000
- **AUC**: 0.625

These metrics can be improved by:
- Expanding the dataset with more transactions
- Engineering additional features
- Fine-tuning model hyperparameters
- Using more sophisticated architectures (GAT, GraphSAGE)

## Usage

### Installation
```bash
pip install torch torch_geometric pandas networkx scikit-learn matplotlib
```

### Running the Model
1. Prepare your transaction data in CSV format with columns: `user_id`, `account_id`, `transaction_id`, `amount`, `fraud`
2. Update the `build_graph` function path to your CSV file
3. Run the notebook cells in sequence

### Expected Output
- Graph visualization
- Training loss at each epoch
- Evaluation metrics (precision, recall, AUC)
- ROC curve plot
- Confusion matrix visualization

## Files

- `fraud_detection_gnn.ipynb`: Complete Jupyter notebook with all code and visualizations
- `README.md`: This documentation file

## Future Improvements

1. **Advanced Architectures**
   - Implement Graph Attention Networks (GAT)
   - Use GraphSAGE for inductive learning
   - Explore temporal graph networks for dynamic transactions

2. **Feature Engineering**
   - Add transaction amount features
   - Incorporate temporal patterns
   - Include merchant and category information

3. **Scalability**
   - Implement mini-batch training
   - Use sampling strategies for large graphs
   - Deploy with distributed training

4. **Real-world Deployment**
   - Create REST API for model inference
   - Implement real-time fraud detection
   - Add model monitoring and drift detection


## Author
Rahul Chandra (Ash-052)

