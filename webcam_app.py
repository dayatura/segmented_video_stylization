from model.transformer_net import TransformerNet
from PIL import Image
from torchvision import models
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import os
import re

# load fcn resnet model for segmentation
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# Define the segmentation function

def segment(net, img):
    trf = T.Compose([T.Resize(256),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    return om

# Get paths and set vars
weights_fname = "candy.pth"                                         # choose candy.pth or green_swirly.pth
script_path = os.path.dirname(os.path.abspath(__file__))
path_to_weights = os.path.join(script_path, "model", weights_fname)
resolution = (256, 256)

# Change to GPU if desired
device = torch.device("cpu")

# Load Style Transfer Model
model = TransformerNet()
with torch.no_grad():
    state_dict = torch.load(path_to_weights)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    model.load_state_dict(state_dict)
    model.to(device)


# Get Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("OpenCV cannot find your webcam! Check that it is under /dev/video0")
    exit(1)

# Capture video and apply segmentation ane stylization
while(True):
    # Grab frame and change to jpeg
    ret, frame = cap.read()
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    img = pil_im.resize(resolution)

    # Transforms image to feed to network
    small_frame_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    small_frame_tensor = small_frame_tensor_transform(img)
    small_frame_tensor = small_frame_tensor.unsqueeze(0).to(device)

    # do stylization
    output = model(small_frame_tensor).cpu()
    styled = output[0]
    styled = styled.clone().clamp(0, 255).detach().numpy()
    styled = styled.transpose(1, 2, 0).astype("uint8")
    
    # do object segmentation 
    om = segment(fcn, img)

    # apply masking using segemtation result

    # ----object dictionary----
    # 0=background 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle 6=bus, 7=car, 
    # 8=cat, 9=chair, 10=cow 11=dining table, 12=dog, 13=horse, 14=motorbike, 
    # 15=person 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    
    object_masks = [15]
    for ob in object_masks:

        idx = om == ob

        b = styled[:,:,0]
        g = styled[:,:,1]
        r = styled[:,:,2]
        img = small_frame_tensor.squeeze()
        b[idx] = img[0,:,:][idx]
        g[idx] = img[1,:,:][idx]
        r[idx] = img[2,:,:][idx]

    # show the frame
    frame = np.stack([r, g, b], axis=2)
    cv2.imshow('Video Stylization -- press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
