import torch.optim as optim
from tqdm import tqdm
# from dataset import ColonoscopyTripletDataset # 导入之前定义的类
# from model import IRModel, GeM # 导入模型相关类
# from losses import TripletLoss # 导入损失函数类

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            # 前向传播
            anchor_desc = model(anchor)
            positive_desc = model(positive)
            negative_desc = model(negative)

            # 计算损失
            loss = criterion(anchor_desc, positive_desc, negative_desc)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    print("Training complete.")

    return model