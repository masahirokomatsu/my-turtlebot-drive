import os, csv, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 15 * 20, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.flatten(self.features(x)))

class OvalDatasetBasic(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        with open(os.path.join(data_dir, 'data.csv')) as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                self.samples.append((row[0], float(row[2])))
        
        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, ang_z = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, rel_path)).convert('RGB')
        return self.transform(img), torch.tensor([ang_z], dtype=torch.float32)

def train():
    data_dir = '/home/ubuntu/dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Training on: {device}')
    
    ds = OvalDatasetBasic(data_dir)
    # num_workers=0 とし、printを挟むことでフリーズを防ぐ
    train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    
    model = BasicNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('[INFO] Starting basic training for 5 epochs...')
    for epoch in range(5):
        model.train()
        t_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        print(f'Epoch {epoch+1}/5 | Train Loss: {t_loss/len(train_loader):.5f}', flush=True)
            
    out_path = '/home/ubuntu/basic_drive_model.pth'
    torch.save(model.state_dict(), out_path)
    print(f'[SUCCESS] Basic Model saved to {out_path}')

if __name__ == '__main__':
    train()
