import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21, 21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
# color = np.random.randint(0, 255, (100, 3))
color = np.zeros((10000, 3))
color[:, 0] = 255

def load_image(imfile):
    img_np = np.array(Image.open(imfile)).astype(np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_t = img_t[None].to(DEVICE)
    return img_t, img_np

def load_image_gray(imfile):
    rgb = np.array(Image.open(imfile)).astype(np.uint8)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray_np = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_np2 = np.expand_dims(gray_np, axis=2)
    gray_t = torch.from_numpy(gray_np2).permute(2, 0, 1).float()
    gray_t = gray_t[None].to(DEVICE)
    gray_np = (np.round(gray_np)).astype(np.uint8)
    # print(gray_np)
    return gray_t, gray_np

def viz(img, flo, kp1, kp2):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # flo_x_y = np.hstack((flo[:,:,0], flo[:,:,1]))
    # plt.imshow(flo_x_y)
    # plt.show()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # print("flo.shape: ", flo.shape)

    # draw the tracks
    img = img.astype(np.uint8)
    mask = np.zeros_like(img.copy())
    for i, (new, old) in enumerate(zip(kp1, kp2)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(img, mask)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_flo = np.concatenate([img, flo], axis=0)
    # cv2.imshow('image', img_flo/255.0)
    # cv2.waitKey()
    
    img_flo = np.concatenate([img, flo], axis=0)
    plt.imshow(img_flo / 255.0)
    plt.show()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1_t, image1_np = load_image(imfile1)
            image2_t, image2_np = load_image(imfile2)
            # print("image1_np.shape: ", image1_np.shape)
            # Get FAST features
            fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
            kp1 = fast.detect(image1_np, None)
            kp1 = cv2.KeyPoint_convert(kp1)
            # print(kp1)


            # Get GFTT
            # feature_params = dict( maxCorners = 100,
            #            qualityLevel = 0.3,
            #            minDistance = 7,
            #            blockSize = 7 )
            # kp1 = cv2.goodFeaturesToTrack(image1_gray_np, mask = None,
            #                  **feature_params)
            # print(kp1)

            # Calculate optical flow
            image1_gray_np = image1_np[:,:,1]
            image2_gray_np = image2_np[:,:,1]
            kp2, st, err = cv2.calcOpticalFlowPyrLK(image1_gray_np, image2_gray_np, kp1, None, **lk_params)
            # Select good points
            # print("kp1.shape: ", kp1.shape)
            # print("st.shape: ", st.shape)
            # print("(st==1).shape: ", (st==1).shape)
            # print("kp1[st==1].shape: ", kp1[st==1].shape)


            padder = InputPadder(image1_t.shape)
            image1_t, image2_t = padder.pad(image1_t, image2_t)

            flow_low, flow_up = model(image1_t, image2_t, iters=20, test_mode=True)
            viz(image1_t, flow_up, kp1, kp2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', default="data_abel", help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

    #1 Default demo: python demo.py --model=models/raft-things.pth --path=demo-frames
    #2 Abel street data: python demo.py --model=models/raft-things.pth --path=data_abel
