from fastsam import FastSAM, FastSAMPrompt
import torch 
import numpy as np
import cv2
import time

model = FastSAM('FastSAM-s.pt')
frame = cv2.imread(IMAGE_PATH)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

everything_results = model(
    source=frame,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
)

prompt_process = FastSAMPrompt(frame.copy(), everything_results, device=DEVICE)
ann = prompt_process.everything_prompt()

img = prompt_process.plot_to_result(ann)

cv2.imshow('Original', frame)
cv2.imshow('Segmented', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
