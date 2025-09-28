# 🐶 DogVision - Dog Breed Classification Model

**Identify and classify 120 different dog breeds with state-of-the-art computer vision accuracy.**

DogVision is a deep learning computer vision model built on ConvNeXt Tiny architecture, specifically fine-tuned for dog breed classification. From Labradors to German Shepherds, this model can accurately identify and classify a wide variety of dog breeds with high precision and fast inference times.

![example2](https://github.com/user-attachments/assets/b7732b52-4781-4ab5-9ddb-b94c53ab985f)

## 🎯 Problem Solved

**Pet Identification Challenge:**
- **Pet owners and veterinarians** often struggle to accurately identify mixed breed dogs or rare breeds
- **Animal shelters** need quick and reliable breed identification for proper care and adoption matching
- **Dog enthusiasts and breeders** require precise breed classification for lineage tracking
- **Mobile and web applications** need lightweight yet accurate breed detection capabilities

**Our Solution:**
- **High-accuracy classification** across 120 distinct dog breeds with ConvNeXt architecture
- **Fast inference times** optimized for real-time applications (sub-second predictions)
- **Robust image handling** that works with various image formats and lighting conditions
- **Production-ready deployment** with user-friendly Gradio interface for immediate use

## ✨ Features

- **120 Dog Breed Classification**: Comprehensive coverage of popular and rare dog breeds
- **ConvNeXt Tiny Architecture**: Modern, efficient CNN with excellent accuracy-to-size ratio
- **Real-Time Inference**: Fast predictions with timing information displayed
- **Top-3 Predictions**: Shows confidence scores for the most likely breeds
- **Image Preprocessing**: Handles various image formats including palette mode images
- **Production Ready**: Compiled model with torch.compile for optimized performance
- **Interactive Interface**: User-friendly Gradio web app with example images

## 🚀 Live Demo

Try the live application: [DogVision on Hugging Face Spaces](https://huggingface.co/spaces/vapit/dogvision)

## 🛠️ Tech Stack

- **Architecture**: ConvNeXt Tiny (pretrained on ImageNet-1K)
- **Framework**: PyTorch with torchvision
- **Optimization**: torch.compile for faster inference
- **Interface**: Gradio with custom styling
- **Image Processing**: Custom transforms with RGBA conversion support
- **Deployment**: Hugging Face Spaces compatible

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dogvision.git
   cd dogvision
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision gradio pandas numpy
   ```

3. **Download model weights:**
   ```bash
   # Ensure convnext_model.pth and labels.csv are in the project directory
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

   The app will be available at `http://localhost:7860`

## 💡 Usage Examples

### Command Line Prediction
```python
from model import create_model
import torch
from PIL import Image

# Load model
model, transforms = create_model(num_classes=120)
model.load_state_dict(torch.load('convnext_model.pth', map_location='cpu'))

# Load and predict
image = Image.open('dog_photo.jpg')
probabilities, inference_time = predict(image)

print(f"Top prediction: {max(probabilities, key=probabilities.get)}")
print(f"Inference time: {inference_time}")
```

### Gradio Interface
```python
import gradio as gr
from app import predict

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Textbox(label="Prediction Time (s)")
    ]
)

demo.launch()
```

## 🏗️ Model Architecture

**Training Pipeline:**
```
Input Image → Preprocessing → ConvNeXt Tiny Backbone → 
Custom Classifier Head → Softmax → Top-K Predictions
```

### Architecture Details
- **Base Model**: ConvNeXt Tiny (28.6M parameters)
- **Input Size**: 224×224 RGB images
- **Backbone**: Frozen ConvNeXt features (transfer learning)
- **Classifier Head**: LayerNorm + Flatten + Linear (768 → 120 classes)
- **Optimization**: torch.compile for inference acceleration

### Custom Preprocessing
```python
# Image preprocessing pipeline
transforms.Compose([
    convert_to_rgba,              # Handle palette mode images
    transforms.Resize((224, 224)), # Resize to model input size
    transforms.ToTensor(),         # Convert to tensor
    ImageNet_normalization        # Apply pretrained normalization
])
```

## 📊 Model Performance

**Classification Capabilities:**
- **Classes Supported**: 120 distinct dog breeds
- **Input Formats**: JPEG, PNG, including palette mode images
- **Inference Speed**: Sub-second predictions on CPU
- **Memory Efficient**: Optimized for deployment environments

**Supported Breeds Include:**
- Popular breeds: Labrador, German Shepherd, Golden Retriever
- Working dogs: Border Collie, Siberian Husky, Rottweiler  
- Small breeds: Chihuahua, Pomeranian, Yorkshire Terrier
- Rare breeds: Azawakh, Telomian, Norwegian Lundehund
- Mixed breed detection capabilities

## 🔧 Technical Features

### Image Handling
- **Format Support**: JPEG, PNG, RGBA, Palette mode
- **Preprocessing**: Automatic RGBA conversion for palette images
- **Resize Handling**: Maintains aspect ratio during preprocessing
- **Memory Optimization**: Efficient tensor operations

### Model Optimization
- **Transfer Learning**: Leverages ImageNet pretrained weights
- **Frozen Backbone**: Prevents overfitting while maintaining features
- **torch.compile**: JIT compilation for faster inference
- **CPU Optimized**: Runs efficiently without GPU requirements

### Production Features
```python
# Key production configurations
model = torch.compile(model)  # Compiled for speed
model.eval()                  # Evaluation mode
torch.inference_mode()        # Disable gradients for inference
```

## 🎨 Gradio Interface Features

- **Image Upload**: Drag-and-drop or click to upload
- **Example Gallery**: Pre-loaded example images for testing
- **Top-3 Predictions**: Shows confidence scores for likely breeds
- **Inference Timing**: Real-time performance metrics
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
dogvision/
├── app.py                 # Gradio interface and prediction logic
├── model.py              # Model architecture and preprocessing
├── convnext_model.pth    # Trained model weights
├── labels.csv            # Breed labels mapping
├── examples/             # Sample images for testing
│   ├── labrador.jpg
│   ├── german_shepherd.jpg
│   └── ...
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Dataset Expansion**: Add more rare breeds or mixed breed detection
2. **Model Architecture**: Experiment with newer vision transformers
3. **Performance Optimization**: Further inference speed improvements
4. **Mobile Deployment**: Optimize for mobile/edge devices
5. **Data Augmentation**: Improve robustness to lighting/angle variations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI Research** for the ConvNeXt architecture
- **PyTorch Team** for the framework and pretrained models
- **Dog breed dataset contributors** for training data
- **Gradio Team** for the intuitive interface framework

## 📞 Contact

Created with 🤎 (and a mixture of mathematics, statistics, and tons of calculations 👩🏽‍🔬) by **Arpit Vaghela**

- GitHub: [@magnifiques](https://github.com/magnifiques)
- Demo: [DogVision on HF Spaces](https://huggingface.co/spaces/vapit/dogvision)
