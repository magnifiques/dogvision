### 1. Imports and class names setup ###
from model import create_model
import pandas as pd
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer
import gradio as gr
import os
import numpy as np

### 1. Create a Dictionary for Dog Breeds
labels_csv = pd.read_csv('./labels.csv')
labels = labels_csv['breed']
labels = np.array(labels)
unique_labels = np.unique(labels)

unique_labels = [' '.join([word.capitalize() for word in label.split('_')]) for label in unique_labels]

### 2. Model and transforms preparation ###
model, model_transforms = create_model(num_classes=len(unique_labels))
model = torch.compile(model)


# Load save weights
model.load_state_dict(torch.load(f='./convnext_model.pth', map_location='cpu',weights_only=True))

# 3. Predict Function

def predict(img) -> Tuple[Dict[str, float], str]:
    """
    Predicts the class probabilities for a given image using a pre-trained model.

    Args:
        img: A PIL image to be predicted.

    Returns:
        A tuple containing:
        - A formatted string displaying class labels and their respective probabilities.
        - The time taken for inference in seconds as a string.
    """
    # Start a timer
    start_time = timer()

    # Put the model into evaluation mode and disable gradient computation
    model.eval()
    with torch.inference_mode():
        
        # Transform the input image for use with the model
        img = model_transforms(img).unsqueeze(dim=0)
        
        # Pass transformed image through the model
        pred_logit = model(img)

    # Turn prediction logits into probabilities
    pred_prob = torch.softmax(pred_logit, dim=1)
    
    pred_label = torch.argmax(pred_prob, dim=1)

    # Map probabilities to class labels
    prediction = unique_labels[pred_label]
    probabilities = {unique_labels[i]: pred_prob[0, i].item() for i in range(len(unique_labels))}

    # Calculate the time taken
    end_time = timer()
    inference_time = end_time - start_time

    # Return predictions as a dictionary and inference time
    return probabilities, f"{inference_time:.4f} seconds"
### 4. Gradio app ###


# Create title, description and article
title = "Dogvision 🐶"
description = "A [ConvNeXt Tiny](https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.convnext_tiny) Computer Vision Model To Classify 120 Dog Breeds 🐩 Ranging fro A Labrador 🐕 to A German Shepherd! 🐕‍🦺"
article = "Created with 🤎 (and a mixture of mathematics, statistics, and tons of calculations 👩🏽‍🔬) by Arpit Vaghela [GitHub](https://github.com/magnifiques)"

# Create example list
example_list = [["./examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[
                    gr.Label(num_top_classes=3, label="Top Predictions"),  # Display top predictions with probabilities
                    gr.Textbox(label="Prediction Time (s)")  # Display inference time
                    ],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=True) # generate a publicly shareable URL
