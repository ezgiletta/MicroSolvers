import requests
from PIL import Image
from torchvision import transforms
import gradio as gr
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


net = torch.load("model.pt")
device = "cuda"
net = net.to(device)

train_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

def predict(inp):
  inp = train_transforms(image=inp)["image"]
  inp = inp.unsqueeze(0)
  inp = inp.to(device)
  #inp = train_transforms(image=inp)["image"]
  with torch.no_grad():
    prediction = net(inp)

  #prediction = {"p_pivot_x": prediction[0][0].item(),
  #"p_pivot_y": prediction[0][1].item(),
  #"p_pivot_z": prediction[0][2].item(),
  #"alpha": prediction[0][3].item(),
  #"beta": prediction[0][4].item(),
  #"gamma": prediction[0][5].item()}
  #print(prediction)
  prediction = prediction.detach().cpu().numpy()
  return list(prediction[0])



gr.Interface(fn=predict,
             inputs=gr.Image(shape=(224, 224)),
             outputs=[gr.Number(label="p_pivot_x"), gr.Number(label="p_pivot_y"), gr.Number(label="p_pivot_z"), gr.Number(label="alpha"), gr.Number(label="beta"), gr.Number(label="gamma")],
             examples=["20221110-134530-315.jpg", "20221110-134532-891.jpg", "20221110-134549-884.jpg"]).launch(share=True)
