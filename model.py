import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Custom transformation to handle palette images
def convert_to_rgba(image):
    # Check if the image mode is 'P' (palette mode)
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image

def create_model(num_classes: int = 120, seed: int = 42):


    # 1. Download the default weights
    weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

    # 2. Setup transforms
    default_transforms = weights.transforms()

    custom_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip images horizontally
    # transforms.Lambda(convert_to_rgba),          # Apply RGBA conversion if necessary
    # transforms.RandomRotation(degrees=10),       # Randomly rotate images by up to 10 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    ])

    # 3. Combine custom and ViT's default transforms
    combined_transforms = transforms.Compose([
      custom_transforms,     # First, apply your custom augmentations
      transforms.Resize((224, 224)),  # Resize to ConvNext's input size if needed (ConvNext expects 224x224)
      transforms.ToTensor(),  # Convert image to Tensor
      default_transforms,         # Apply default normalization (mean, std)
     ])

    # 4. Create a model and apply the default weights
    model = torchvision.models.convnext_tiny(weights=weights)

    # 5. Freeze the base layers in the model (this will stop all layers from training)
    for parameters in model.parameters():
        parameters.requires_grad = False

    # 6. Set seeds for reproducibility
    torch.manual_seed(seed)

    # 7. Modify the number of output layers (add a dropout layer for regularization)
    model.classifier = nn.Sequential(
      nn.LayerNorm([768, 1, 1], eps=1e-06, elementwise_affine=True),  # Apply LayerNorm on the channel dimension (768)
      nn.Flatten(start_dim=1),  # Flatten the tensor from dimension 1 onwards (batch size remains intact)
      nn.Linear(in_features=768, out_features=num_classes, bias=True)
    )


    return model, combined_transforms
