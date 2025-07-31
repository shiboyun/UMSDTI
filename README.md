# UMSDTI: Unifying Multiple Molecular and Sequence Perspectives for Drug–Target Interaction Prediction

## Overview
UMSDTI is a deep learning framework for Drug–Target Interaction (DTI) prediction. It employs a multi-view co-attention architecture that captures drug information from multiple chemical perspectives and protein information from sequence and residue-level interactions, offering a comprehensive and interpretable modeling pipeline.

## Key Contributions
- **Multi-Modal Drug Encoding**: UMSDTI extracts drug features from SMILES sequences, atomic graphs, and chemical bonds, overcoming limitations of traditional GNNs in capturing positional information and enhancing chemical expressiveness.
- **Structure-Free Protein Contact Modeling**: Instead of relying on 3D structural data, UMSDTI leverages attention maps from pretrained protein language models to infer contact patterns, complementing sequential protein representations.
- **Cross-Modal Interaction Mechanism**: A co-attention module facilitates multi-granular interactions between drug and target representations, offering interpretability and biological insights into DTI mechanisms.
- **State-of-the-Art Performance**: Extensive experiments demonstrate that UMSDTI outperforms eight cutting-edge methods across multiple benchmark datasets. Visualizations of attention weights and newly discovered DTIs further highlight its effectiveness and interpretability.

## System Requirements
- **Python**: 3.9+
- **CUDA**: 11.8+ (recommended)
- **Memory**: 16 GB+
- **GPU**: 24 GB+ VRAM

## Dependencies
Install the following packages (either manually or using `requirements.txt`):

```bash
# Core deep learning and scientific packages
torch==2.5.1
transformers==4.24.0
rdkit-pypi
chemprop
numpy==1.23.0
pandas==2.2.3
scikit-learn>=1.5.2
tqdm==4.67.0
prefetch-generator
easydict
pyyaml
```

## Installation Guide

### 1. Clone the repository
```bash
cd UMSDTI
```

### 2. Create and activate a virtual environment
```bash
conda create -n umsdti python=3.9
conda activate umsdti
```

### 3. Install PyTorch (with CUDA 11.8)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install PyG (PyTorch Geometric)
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1%2Bcu118.html
```

### 5. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 6. Download pretrained models
Create the model directory:
```bash
mkdir -p PretrainedModels/
```
Download the ESM model using Transformers or from Meta ESM GitHub.
Set the `--esm_model_path` in `main_drugbank.py` to your local model directory.

## Dataset Format
Each line of the dataset should contain:
```
<SMILES> <Protein Sequence> <Label (0/1)>
```
**Example:**
```
C[C@H](C1=CC=C(C=C1)Cl)... MSGPVPSRARVYTDVNTHRPREYWDYESHVVEW... 1
```

## Usage
### 1. Train on the DrugBank dataset
```bash
python main_drugbank.py
```

## Key Hyperparameters

### Training Parameters
| Argument         | Description                | Default |
|------------------|---------------------------|---------|
| --seed           | Random seed               | 42      |
| --Batch_size     | Batch size                | 16      |
| --Epoch          | Number of epochs          | 100     |
| --device         | Training device           | cuda    |
| --weight_decay   | Weight decay for optimizer| 1e-4    |
| --ema_decay      | Exponential moving average decay | 0.997 |

### Learning Rates
| Component         | Argument                        | Default |
|-------------------|----------------------------------|---------|
| Drug encoder      | --drug_encoder_learning_rate     | 2e-4    |
| Protein encoder   | --protein_encoder_learning_rate  | 2e-4    |
| Interaction module| --interaction_learning_rate      | 1e-4    |

### Model Architecture
| Argument             | Description                        | Default |
|----------------------|-------------------------------------|---------|
| --hidden_size        | Hidden dimension size               | 128     |
| --protein_hidden_size| Protein hidden dimension            | 128     |
| --depth              | GNN message passing steps           | 4       |
| --dropout            | Dropout rate                        | 0.1     |
| --protein_max_length | Max sequence length for proteins    | 2048    |

## Model Architecture

1. **Drug Encoder**
   - **MPN (Message Passing Network)**: Encodes atomic-level graph features.
   - **CNN for SMILES**: Captures sequential substructure patterns.

2. **Protein Encoder**
   - **ESM Transformer**: Leverages pretrained protein embeddings.
   - **CNN Layers**: Extract local residue features from sequences.

3. **Interaction Module**
   - **Cross-Attention**: Aligns and fuses drug–protein features at multiple levels.

4. **Classifier**
   - **MLP (Multi-Layer Perceptron)**: Outputs DTI probability scores via Softmax.

---
