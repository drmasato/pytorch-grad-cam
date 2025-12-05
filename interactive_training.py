import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
import sys

def load_or_create_model(model_path, device):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # 2 classes: Normal, Pneumonia
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Creating new model (pretrained ResNet50)")
    
    model = model.to(device)
    return model

def train_one_image(model, image_path, label, optimizer, criterion, device):
    model.train()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        target = torch.tensor([label]).to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Get prediction
        _, pred = torch.max(output, 1)
        return loss.item(), pred.item()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def main():
    model_path = 'medical_resnet50.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_or_create_model(model_path, device)
    criterion = nn.CrossEntropyLoss()
    # Use a small learning rate for fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("\n--- Interactive Training Mode ---")
    print("Enter 'q' to quit at any time.")
    print("Labels: 0 = Normal, 1 = Pneumonia")
    
    while True:
        print("\n" + "-"*30)
        image_path = input("Enter image path: ").strip()
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print("File not found. Please try again.")
            continue
            
        label_str = input("Enter label (0 for Normal, 1 for Pneumonia): ").strip()
        if label_str.lower() == 'q':
            break
            
        if label_str not in ['0', '1']:
            print("Invalid label. Please enter 0 or 1.")
            continue
            
        label = int(label_str)
        
        print(f"Training on {image_path} with label {label}...")
        loss, pred = train_one_image(model, image_path, label, optimizer, criterion, device)
        
        if loss is not None:
            print(f"Training step complete.")
            print(f"Loss: {loss:.4f}")
            print(f"Model prediction before update was: {'Normal' if pred == 0 else 'Pneumonia'}")
            
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
