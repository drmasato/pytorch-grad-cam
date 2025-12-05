import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
import json
import glob
from datetime import datetime

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

def get_prediction(model, image_path, device, class_names):
    model.eval()
    
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
            _, pred = torch.max(output, 1)
            confidence = probabilities[pred].item()
        
        return pred.item(), confidence
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None, None

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
        
        return loss.item()
    except Exception as e:
        print(f"Error training on image: {e}")
        return None

def display_image(image_path, pred_label=None, confidence=None, class_names=None):
    plt.figure(figsize=(10, 8))
    img = mpimg.imread(image_path)
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
    title = f"Image: {os.path.basename(image_path)}"
    if pred_label is not None and class_names is not None:
        title += f"\nModel Prediction: {class_names[pred_label]} (confidence: {confidence:.2%})"
    
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def save_progress(progress_file, processed_images, labels):
    with open(progress_file, 'w') as f:
        json.dump({
            'processed_images': processed_images,
            'labels': labels,
            'last_updated': datetime.now().isoformat()
        }, f, indent=2)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get('processed_images', []), data.get('labels', {})
    return [], {}

def main():
    parser = argparse.ArgumentParser(description='Batch Annotation and Training')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing images')
    parser.add_argument('--classification_type', type=str, default='pattern',
                        choices=['features', 'pattern'],
                        help='Type of classification')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model version every N images')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous session')
    
    args = parser.parse_args()
    
    # Setup
    classification_type = args.classification_type
    class_info = CLASSIFICATION_TYPES[classification_type]
    num_classes = len(class_info['classes'])
    class_names = [c.split(': ', 1)[1] for c in class_info['classes']]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classification Type: {class_info['name']}")
    print(f"Number of classes: {num_classes}\n")
    
    # Get image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.folder, ext)))
        image_files.extend(glob.glob(os.path.join(args.folder, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images in {args.folder}\n")
    
    if len(image_files) == 0:
        print("No images found. Exiting.")
        return
    
    # Setup model and progress tracking
    model_base_path = f'medical_resnet50_{classification_type}'
    current_model_path = f'{model_base_path}_current.pth'
    progress_file = f'{model_base_path}_progress.json'
    
    model = load_or_create_model(current_model_path, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Load progress if resuming
    processed_images, labels_dict = [], {}
    if args.resume:
        processed_images, labels_dict = load_progress(progress_file)
        print(f"Resuming: {len(processed_images)} images already processed\n")
    
    # Filter out already processed images
    remaining_images = [img for img in image_files if img not in processed_images]
    
    print(f"Images to annotate: {len(remaining_images)}")
    print(f"\nClass options:")
    for class_label in class_info['classes']:
        print(f"  {class_label}")
    print("\nCommands: 's' = skip, 'q' = quit and save, 'help' = show classes\n")
    
    version_counter = len(processed_images) // args.save_interval
    
    try:
        for idx, image_path in enumerate(remaining_images):
            print(f"\n{'='*60}")
            print(f"Progress: {len(processed_images) + idx + 1}/{len(image_files)}")
            print(f"{'='*60}")
            
            # Get model prediction
            pred_label, confidence = get_prediction(model, image_path, device, class_names)
            
            # Display image
            display_image(image_path, pred_label, confidence, class_names)
            
            # Get user input
            while True:
                label_input = input(f"\nEnter label (0-{num_classes-1}, 's' to skip, 'q' to quit, 'help' for classes): ").strip()
                
                if label_input.lower() == 'q':
                    print("\nQuitting and saving progress...")
                    plt.close('all')
                    save_progress(progress_file, processed_images, labels_dict)
                    torch.save(model.state_dict(), current_model_path)
                    print(f"Progress saved to {progress_file}")
                    print(f"Model saved to {current_model_path}")
                    return
                
                if label_input.lower() == 's':
                    print("Skipping this image.")
                    plt.close()
                    break
                
                if label_input.lower() == 'help':
                    print("\nClass options:")
                    for class_label in class_info['classes']:
                        print(f"  {class_label}")
                    continue
                
                try:
                    label = int(label_input)
                    if 0 <= label < num_classes:
                        # Train on this image
                        print(f"Training on label: {class_names[label]}...")
                        loss = train_one_image(model, image_path, label, optimizer, criterion, device)
                        
                        if loss is not None:
                            print(f"✓ Training complete. Loss: {loss:.4f}")
                            
                            # Update progress
                            processed_images.append(image_path)
                            labels_dict[image_path] = label
                            
                            # Save progress
                            save_progress(progress_file, processed_images, labels_dict)
                            
                            # Check if we should save a version
                            if len(processed_images) % args.save_interval == 0:
                                version_counter += 1
                                version_path = f'{model_base_path}_v{version_counter}.pth'
                                torch.save(model.state_dict(), version_path)
                                print(f"✓ Model version saved: {version_path}")
                            
                            # Save current model
                            torch.save(model.state_dict(), current_model_path)
                        
                        plt.close()
                        break
                    else:
                        print(f"Invalid label. Please enter 0-{num_classes-1}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        print(f"\n{'='*60}")
        print("All images processed!")
        print(f"Total annotated: {len(processed_images)}")
        print(f"Final model saved to: {current_model_path}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        save_progress(progress_file, processed_images, labels_dict)
        torch.save(model.state_dict(), current_model_path)
        print(f"Progress saved to {progress_file}")
        print(f"Model saved to {current_model_path}")
    
    plt.close('all')

if __name__ == '__main__':
    main()
