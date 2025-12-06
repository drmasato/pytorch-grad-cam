import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def get_user_choice(options, prompt_text):
    print(f"\n{prompt_text}")
    for i, option in enumerate(options):
        print(f"{i + 1}: {option}")
    
    while True:
        try:
            choice = input(f"Select (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

def playground():
    """
    Grad-CAM 実験用プレイグラウンド (Interactive Version)
    """
    
    print("\n" + "="*40)
    print("      Grad-CAM Playground")
    print("="*40)

    # ---------------------------------------------------------
    # STEP 1: モデルを選ぶ (今のところResNet50固定)
    # ---------------------------------------------------------
    print("\n[STEP 1] Model Selection")
    print("Using ResNet50 (pretrained) for this demo.")
    # 将来的にはここも選択可能にできます
    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    # ---------------------------------------------------------
    # STEP 2: 画像を選ぶ
    # ---------------------------------------------------------
    image_options = [
        'examples/dog.jpg',
        'examples/cat.jpg',
        'examples/horses.jpg',
        'examples/dog_cat.jfif'
    ]
    
    # ユーザーが独自の画像パスを入力したい場合のオプションを追加
    image_choice_str = get_user_choice(image_options + ["Enter custom path"], "[STEP 2] Select an Image")
    
    if image_choice_str == "Enter custom path":
        image_path = input("Enter image path: ").strip()
    else:
        image_path = image_choice_str

    print(f"Loading image from {image_path}...")
    
    try:
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please check if the image path is correct.")
        return

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # ---------------------------------------------------------
    # STEP 3: 手法を選ぶ
    # ---------------------------------------------------------
    method_options = [
        "gradcam", "hirescam", "scorecam", "gradcam++", 
        "ablationcam", "xgradcam", "eigencam", "fullgrad"
    ]
    
    method_name = get_user_choice(method_options, "[STEP 3] Select CAM Method")
    
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "fullgrad": FullGrad
    }

    print(f"Creating {method_name}...")
    cam = methods[method_name](model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # ---------------------------------------------------------
    # STEP 4: 実行と保存
    # ---------------------------------------------------------
    targets = None 

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    output_path = f'playground_{method_name}.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("\n" + "-"*40)
    print(f"Done! Result saved to: {output_path}")
    print("-"*40 + "\n")

if __name__ == '__main__':
    playground()
