import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import argparse

CLASSIFICATION_TYPES = {
    'pattern': {
        'name': 'ILD Imaging Pattern (画像パターン)',
        'classes': [
            'Normal (正常)',
            'UIP/P',
            'NSIP/P',
            'BIP',
            'COP',
            'RBILD',
            'MIP/DIP',
            'PPFE',
            'AIP',
            'Unclassifiable (分類不能)'
        ]
    }
}

def load_model(model_path, num_classes, device, is_pretrained_only=False):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if not is_pretrained_only and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Using pretrained ResNet50 (not fine-tuned)")
    
    model = model.to(device)
    model.eval()
    return model

def get_prediction(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        return top3_idx.cpu().numpy(), top3_prob.cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Compare original vs trained model')
    parser.add_argument('--folder', type=str, required=True, help='Folder with test images')
    parser.add_argument('--trained_model', type=str, 
                        default='medical_resnet50_pattern_current.pth',
                        help='Path to trained model')
    parser.add_argument('--classification_type', type=str, default='pattern')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    class_info = CLASSIFICATION_TYPES[args.classification_type]
    num_classes = len(class_info['classes'])
    class_names = class_info['classes']
    
    # Load both models
    print("Loading original pretrained model...")
    original_model = load_model(args.trained_model, num_classes, device, is_pretrained_only=True)
    
    print("Loading trained model...")
    trained_model = load_model(args.trained_model, num_classes, device, is_pretrained_only=False)
    
    # Get test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.folder, ext)))
        image_files.extend(glob.glob(os.path.join(args.folder, ext.upper())))
    
    image_files = sorted(list(set(image_files)))[:10]  # Limit to 10 for display
    
    print(f"\nComparing predictions on {len(image_files)} images:\n")
    print("="*100)
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Get predictions from both models
        orig_idx, orig_prob = get_prediction(original_model, img_path, device)
        trained_idx, trained_prob = get_prediction(trained_model, img_path, device)
        
        if orig_idx is None or trained_idx is None:
            continue
        
        print(f"\n📷 {img_name}")
        print("-" * 100)
        
        print("  Original Model (未学習):")
        for i in range(3):
            print(f"    {i+1}. {class_names[orig_idx[i]]}: {orig_prob[i]:.2%}")
        
        print("\n  Trained Model (学習後):")
        for i in range(3):
            marker = "✓" if i == 0 else " "
            print(f"  {marker} {i+1}. {class_names[trained_idx[i]]}: {trained_prob[i]:.2%}")
        
        # Highlight if prediction changed
        if orig_idx[0] != trained_idx[0]:
            print(f"\n  ⚠️  Prediction changed: {class_names[orig_idx[0]]} → {class_names[trained_idx[0]]}")
        else:
            print(f"\n  ✓ Prediction unchanged: {class_names[trained_idx[0]]}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
