import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel, RobertaTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from config import (
    FEATURES_DIR, 
    ROBERTA_MODEL_DIR, 
    TRAIN_PATH, 
    VAL_PATH, 
    TEST_PATH, 
    NUM_LABELS,
    MODEL_DIR
)

class GatedFusionNetwork(nn.Module):
    def __init__(self, roberta_dim=768, tfidf_dim=5000, hidden_dim=512, num_labels=NUM_LABELS):
        super(GatedFusionNetwork, self).__init__()
        
        self.tfidf_projection = nn.Sequential(
            nn.Linear(tfidf_dim, roberta_dim),
            nn.BatchNorm1d(roberta_dim),
            nn.ReLU()
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(roberta_dim * 2, roberta_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(roberta_dim, num_labels)

    def forward(self, roberta_emb, tfidf_vec, return_features=False):
        tfidf_proj = self.tfidf_projection(tfidf_vec)
        
        concat_features = torch.cat([roberta_emb, tfidf_proj], dim=1)
        z = self.gate_net(concat_features)
        
        fused_features = (z * roberta_emb) + ((1 - z) * tfidf_proj)
        
        if return_features:
            return fused_features
        
        logits = self.classifier(fused_features)
        return logits

    def get_gate_values(self, roberta_emb, tfidf_vec):
        tfidf_proj = self.tfidf_projection(tfidf_vec)
        concat_features = torch.cat([roberta_emb, tfidf_proj], dim=1)
        z = self.gate_net(concat_features)
        return z


def load_tfidf_features(split_name):
    path = os.path.join(FEATURES_DIR, "tfidf", f"tfidf_{split_name}_features.csv")
    print(f"Loading TF-IDF: {path}")
    df = pd.read_csv(path)
    return df.values.astype(np.float32)

def load_labels():
    """Load labels from original processed CSVs"""
    print("Loading labels...")
    y_train = pd.read_csv(TRAIN_PATH)["class_id"].values
    y_val = pd.read_csv(VAL_PATH)["class_id"].values
    y_test = pd.read_csv(TEST_PATH)["class_id"].values
    return y_train, y_val, y_test

def get_roberta_embeddings(split_name, model, batch_size=32, device='cuda'):
    """Extract CLS embeddings from fine-tuned RoBERTa"""
    tokenized_path = os.path.join(FEATURES_DIR, "roberta")
    dataset = load_from_disk(tokenized_path)[split_name]
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    model.to(device)
    
    embeddings = []
    print(f"Extracting RoBERTa embeddings for {split_name}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu().numpy())
            
    return np.vstack(embeddings)

def train_fusion_model(model, train_loader, val_loader, device, epochs=10):
    """Train the fusion network to learn the gate weights"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    
    best_val_acc = 0.0
    
    print("\n--- Starting Fusion Model Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for r_batch, t_batch, y_batch in train_loader:
            r_batch, t_batch, y_batch = r_batch.to(device), t_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(r_batch, t_batch) 
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for r_batch, t_batch, y_batch in val_loader:
                r_batch, t_batch, y_batch = r_batch.to(device), t_batch.to(device), y_batch.to(device)
                logits = model(r_batch, t_batch)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                truths.extend(y_batch.cpu().numpy())
        
        val_acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        model_path = f"{MODEL_DIR}/fusion_model.path"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            
    print("Training finished. Loading best model...")
    model.load_state_dict(torch.load("best_fusion_model.pth"))
    return model

def extract_and_save_fused(model, roberta_data, tfidf_data, split_name, device):
    model.eval()
    model.to(device)
    
    dataset = TensorDataset(torch.tensor(roberta_data), torch.tensor(tfidf_data))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    fused_list = []
    print(f"Generating fused features for {split_name}...")
    
    with torch.no_grad():
        for r_batch, t_batch in tqdm(loader):
            r_batch, t_batch = r_batch.to(device), t_batch.to(device)
            
            features = model(r_batch, t_batch, return_features=True)
            fused_list.append(features.cpu().numpy())
            
    fused_array = np.vstack(fused_list)
    
    save_dir = os.path.join(FEATURES_DIR, "gated_fusion")
    os.makedirs(save_dir, exist_ok=True)
    
    cols = [str(i) for i in range(fused_array.shape[1])]
    df = pd.DataFrame(fused_array, columns=cols)
    
    save_path = os.path.join(save_dir, f"gated_{split_name}_features.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved fused features to {save_path}")

def print_weight_analysis(model, roberta_data, tfidf_data, device, num_samples=5):
    print("\n" + "="*50)
    print(" Analyzing fusion weight")
    print("="*50)
    
    model.eval()
    indices = np.random.choice(len(roberta_data), num_samples, replace=False)
    
    r_sample = torch.tensor(roberta_data[indices]).to(device)
    t_sample = torch.tensor(tfidf_data[indices]).to(device)
    
    with torch.no_grad():
        z_values = model.get_gate_values(r_sample, t_sample)
        
        z_mean = z_values.mean(dim=1).cpu().numpy()
    
    print(f"{'Sample ID':<10} | {'RoBERTa Weight (z)':<20} | {'TF-IDF Weight (1-z)':<20} | {'Decision'}")
    print("-" * 70)
    
    avg_r = []
    for i, idx in enumerate(indices):
        r_w = z_mean[i]
        t_w = 1.0 - r_w
        avg_r.append(r_w)
        decision = "RoBERTa" if r_w > 0.5 else "TF-IDF"
        print(f"{idx:<10} | {r_w:.4f}{' '*14} | {t_w:.4f}{' '*14} | {decision}")
    
    print("-" * 70)
    mean_r = np.mean(avg_r)
    print(f"Average Batch Preference: {mean_r:.4f} -> Model leans towards {'RoBERTa' if mean_r > 0.5 else 'TF-IDF'}")
    print("="*50 + "\n")
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    y_train, y_val, y_test = load_labels()

    tfidf_train = load_tfidf_features("train")
    tfidf_val = load_tfidf_features("val")
    tfidf_test = load_tfidf_features("test")

    print("\nLoading RoBERTa model to extract embeddings...")
    roberta_base = RobertaModel.from_pretrained(ROBERTA_MODEL_DIR) 
    
    roberta_train = get_roberta_embeddings("train", roberta_base, device=device)
    roberta_val = get_roberta_embeddings("validation", roberta_base, device=device)
    roberta_test = get_roberta_embeddings("test", roberta_base, device=device)

    train_dataset = TensorDataset(
        torch.tensor(roberta_train), 
        torch.tensor(tfidf_train), 
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(roberta_val), 
        torch.tensor(tfidf_val), 
        torch.tensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    fusion_model = GatedFusionNetwork(
        roberta_dim=768, 
        tfidf_dim=tfidf_train.shape[1], 
        num_labels=NUM_LABELS
    )
    fusion_model.to(device)

    trained_model = train_fusion_model(fusion_model, train_loader, val_loader, device, epochs=15)

    print_weight_analysis(trained_model, roberta_val, tfidf_val, device, num_samples=10)
    
    print("\nExtracting Final Fused Features...")
    extract_and_save_fused(trained_model, roberta_train, tfidf_train, "train", device)
    extract_and_save_fused(trained_model, roberta_val, tfidf_val, "val", device)
    extract_and_save_fused(trained_model, roberta_test, tfidf_test, "test", device)

    fusion_model_dir = os.path.join(MODEL_DIR, "gated_fusion")
    os.makedirs(fusion_model_dir, exist_ok=True)
    
    final_model_path = os.path.join(fusion_model_dir, "fusion_network.pth")
    
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"\nSaved Gated Fusion model to: {final_model_path}")
        
    print("\nCombining features finished.")

if __name__ == "__main__":
    main()