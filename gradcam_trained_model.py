import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSIFICATION_TYPES = {
    'pattern': {
        'name': 'ILD Imaging Pattern',
        'classes': [
            'Normal', 'UIP/P', 'NSIP/P', 'BIP', 'COP',
            'RBILD', 'MIP/DIP', 'PPFE', 'AIP', 'Unclassifiable'
        ]
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--model-path', type=str,
                        default='medical_resnet50_pattern_current.pth',
                        help='Path to trained model')
    parser.add_argument('--classification-type', type=str, default='pattern',
                        choices=['pattern'],
                        help='Classification type')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle component')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam', 'ablationcam',
                                 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'],
                        help='CAM method to use')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    args = get_args()
    
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    # Load class information
    class_info = CLASSIFICATION_TYPES[args.classification_type]
    num_classes = len(class_info['classes'])
    class_names = class_info['classes']

    # Load model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Loading trained model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location='cuda' if args.use_cuda else 'cpu'))
    
    if args.use_cuda:
        model = model.cuda()
    
    model.eval()

    target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    print("\nModel Predictions:")
    for i in range(3):
        print(f"  {i+1}. {class_names[top3_idx[i]]}: {top3_prob[i].item():.2%}")
    
    # Use top prediction as target
    targets = [ClassifierOutputTarget(top3_idx[0].item())]
    print(f"\nGenerating CAM for: {class_names[top3_idx[0]]}\n")

    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=top3_idx[0].item())

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    output_name = f'trained_{args.method}_cam.jpg'
    cv2.imwrite(output_name, cam_image)
    cv2.imwrite(f'trained_{args.method}_gb.jpg', gb)
    cv2.imwrite(f'trained_{args.method}_cam_gb.jpg', cam_gb)
    
    print(f"Saved: {output_name}")
    print(f"Saved: trained_{args.method}_gb.jpg")
    print(f"Saved: trained_{args.method}_cam_gb.jpg")
