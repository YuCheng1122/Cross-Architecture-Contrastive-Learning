import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss

def load_data(path, batch_size=64):
    with open(path, "rb") as f:
        data = pickle.load(f)  # list of (vec1, vec2, label)

    v1 = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    labels = torch.tensor(np.array([d[2] for d in data]), dtype=torch.long)

    dataset = TensorDataset(v1, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (vectors, labels) in enumerate(dataloader):
        vectors, labels = vectors.to(device), labels.to(device)
        
        features = model(vectors)
        loss = criterion(features, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def main():
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 超參數
    INPUT_DIM = 256
    HIDDEN_DIM = 256
    OUTPUT_DIM = 256
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 50
    
    # 載入資料（30% 負樣本）
    train_loader = load_data("/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector_Os.pickle", BATCH_SIZE, negative_ratio=0.3)
    
    # 建立模型
    model = ProjectionHead(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = SupervisedContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 訓練
    print("開始訓練...")
    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}')
        
        # 可選：儲存 checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    # 保存最終模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("訓練完成!")

if __name__ == "__main__":
    main()