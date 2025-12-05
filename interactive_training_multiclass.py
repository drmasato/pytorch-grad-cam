import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
import argparse

# Classification types and their classes
CLASSIFICATION_TYPES = {
    'features': {
        'name': 'Imaging Features (画像所見)',
        'classes': [
            '0: Normal (正常)',
            '1: Reticular Pattern (網状影)',
            '2: Ground Glass Opacity (すりガラス影)',
            '3: Bronchiectasis (気管支拡張)',
            '4: Centrilobular (小葉中心性)',
            '5: Honeycombing (蜂巣肺)'
        ]
    },
    'pattern': {
        'name': 'ILD Imaging Pattern (画像パターン)',
        'classes': [
            '0: Normal (正常)',
            '1: UIP/P (Usual Interstitial Pneumonia Pattern)',
            '2: NSIP/P (Nonspecific Interstitial Pneumonia Pattern)',
            '3: BIP (Bronchiolitis Interstitial Pneumonia)',
            '4: COP (Cryptogenic Organizing Pneumonia)',
            '5: RBILD (Respiratory Bronchiolitis-ILD)',
            '6: MIP/DIP (Macrophage Interstitial Pneumonia)',
            '7: PPFE (Pleuroparenchymal Fibroelastosis)',
            '8: AIP (Acute Interstitial Pneumonia)',
            '9: Unclassifiable (分類不能)'
        ]
    }
}

def load_or_create_model(model_path, num_classes, device):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Starting with fresh model.")
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
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = torch.max(output, 1)
        confidence = probabilities[pred].item()
        
        return loss.item(), pred.item(), confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Interactive Multi-Class ILD Training')
    parser.add_argument('--classification_type', type=str, default='pattern',
                        choices=['features', 'pattern'],
                        help='Type of classification: features or pattern')
    args = parser.parse_args()
    
    classification_type = args.classification_type
    class_info = CLASSIFICATION_TYPES[classification_type]
    num_classes = len(class_info['classes'])
    
    model_path = f'medical_resnet50_{classification_type}.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nClassification Type: {class_info['name']}")
    print(f"Number of classes: {num_classes}")
    
    model = load_or_create_model(model_path, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("\n--- Interactive Training Mode ---")
    print("Enter 'q' to quit at any time.")
    print("Enter 'help' to see class list.\n")
    
    while True:
        print("\n" + "="*50)
        command = input("Enter command (image path, 'help', or 'q'): ").strip()
        
        if command.lower() == 'q':
            break
        
        if command.lower() == 'help':
            print(f"\n{class_info['name']} Classes:")
            for class_label in class_info['classes']:
                print(f"  {class_label}")
            continue
        
        image_path = command
        if not os.path.exists(image_path):
            print("File not found. Please try again.")
            continue
        
        print(f"\n{class_info['name']} Classes:")
        for class_label in class_info['classes']:
            print(f"  {class_label}")
            
        label_str = input("\nEnter label number: ").strip()
        if label_str.lower() == 'q':
            break
            
        try:
            label = int(label_str)
            if label < 0 or label >= num_classes:
                print(f"Invalid label. Please enter a number between 0 and {num_classes-1}.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        print(f"\nTraining on {image_path} with label {label}...")
        loss, pred, confidence = train_one_image(model, image_path, label, optimizer, criterion, device)
        
        if loss is not None:
            print(f"\n✓ Training step complete.")
            print(f"  Loss: {loss:.4f}")
            print(f"  Model prediction: {class_info['classes'][pred]} (confidence: {confidence:.2%})")
            
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved to {model_path}")

if __name__ == '__main__':
    main()
