
from matplotlib.pyplot import axis
import gradio as gr
import requests
import numpy as np
from torch import nn
import cv2
import requests
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import os
from PIL import Image

car_metadata = MetadataCatalog.get("my_dataset_val")
car_metadata.thing_classes = ['Damages', 'Dent', 'Dislocation', 'Scratch', 'Shatter'] 
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file("myconfig2.yml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.WEIGHTS = "model_final.pth"
predictor = DefaultPredictor(cfg)
def inference(img):
    im = cv2.imread(img.name)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],metadata=car_metadata , scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return Image.fromarray(np.uint8(out.get_image())).convert('RGB')
    

title = "Detectron2 Car Damage Detection 🚗"
description = "An Model which detects the Damage on car and classifies as Dents,Scratches,Dislocation and Shatter."
article = "Created by Vishal Jadhav (www.linkedin.com/in/vishaljadhav1855)"
examples = [['29.jpg','122.jpg','68.jpg']]
gr.Interface(inference, inputs=gr.inputs.Image(type="file"), outputs=gr.outputs.Image(type="pil"),enable_queue=True, title=title,
    description=description,
    article=article,
    examples=examples).launch()