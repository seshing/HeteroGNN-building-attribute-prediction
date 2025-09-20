import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import Data as HomoData  # Adjust import if needed
from transformers import AutoImageProcessor, AutoModel

class ImageDataset(Dataset):

    def __init__(self, root_dir, bids, transform=None):
        self.root_dir = root_dir
        self.bids = bids
        self.transform = transform

    def __len__(self):
        return len(self.bids)

    def __getitem__(self, idx):
        # get building ID and its file
        bid = self.bids[idx]
        img_path = os.path.join(self.root_dir, f"{bid}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, bid

class DinoV3Embedder(nn.Module):
    def __init__(self, pretrained_model_name, output_dimension):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True
        )
        
        # Define the projection head to get the desired output dimension
        self.projection = nn.Linear(self.backbone.config.hidden_size, output_dimension)

    def forward(self, pixel_values):
        """
        The forward pass that takes preprocessed pixel values and returns embeddings.
        """
        # 1. Get the raw outputs from the backbone
        outputs = self.backbone(pixel_values=pixel_values)
        
        # 2. Extract the pooled output (embedding for the [CLS] token)
        pooled_output = outputs.pooler_output
        
        # 3. Project to the desired dimension
        embedding = self.projection(pooled_output)
        
        return embedding

def process_svi_images(task, model_name, image_path, all_bids_graph, bid2node, image_dimension, batch_size, device, output_path, city):
    pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = DinoV3Embedder(pretrained_model_name, image_dimension)
    transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    model = model.to(device).eval()
    
    available = [bid for bid in all_bids_graph if os.path.isfile(os.path.join(image_path, f"{bid}.png"))]
    print(f"Found {len(available)} available SVI images for {task}.")

    cache_file = f"{city}_{task}_{image_dimension}_{model_name}_{len(all_bids_graph)}_{len(available)}.pt"
    cache_path = os.path.join(output_path, cache_file)

    ds = ImageDataset(root_dir=image_path, bids=available, transform=transform)

    # Cache & embed
    image_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    if os.path.exists(cache_path):
        image_feat = torch.load(cache_path, map_location=device)
        print(f"Cache file loaded: {cache_file}")
    else:
        print(f"Cache file not found. Proceeding with embedding {task}...")
        image_feat = torch.full((len(all_bids_graph), image_dimension), float('nan'), device=device)
        with torch.no_grad():
            for imgs, bids in tqdm(image_loader, desc="Embedding SVI"):
                imgs = imgs.to(device)
                try:
                    embs = model(imgs)
                    for emb, bid in zip(embs, bids):
                        node_idx = bid2node[str(bid)]
                        image_feat[node_idx] = emb
                except Exception as e:
                    print(f"Error processing image for bid {bid}: {e}. Skipping...")

        torch.save(image_feat, cache_path)
        print(f"Cache file saved as: {cache_file}")

    return image_feat


def build_knn_graph(graph_model, data, hidden_dim, device, k=5, chunk_size=2048):
    graph_model = graph_model.to(device)
    data = data.to(device)
    graph_model.eval()
    with torch.no_grad():
        E = graph_model.embed(data.x_dict, data.edge_index_dict)

    E_cpu  = E.cpu().numpy()
    E_norm = F.normalize(torch.from_numpy(E_cpu), p=2, dim=1).numpy()  # [N, d]

    n_jobs = os.cpu_count()
    nn     = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=n_jobs)
    nn.fit(E_norm)

    rows, cols = [], []
    for start in tqdm(range(0, E_norm.shape[0], chunk_size), desc="Building k-NN graph"):
        end    = min(start + chunk_size, E_norm.shape[0])
        chunk  = E_norm[start:end]                 # [B, d]
        dists, idx = nn.kneighbors(chunk)          # [B, k+1]
        idx       = idx[:, 1:]                     # drop self → [B, k]
        B         = end - start
        rows.append(np.repeat(np.arange(start, end), k))
        cols.append(idx.reshape(-1))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    edge_index_vis = torch.tensor([rows, cols], dtype=torch.long)
    homo = HomoData(edge_index=edge_index_vis.to(device))
    return homo