import matplotlib.pyplot as plt
import numpy as np
import math


def addnoise(img,sigma):
    # convert image to numpy array
    data = np.asarray(img)
    # convert to float
    data = np.float32(data)
    row, col = data.shape
    noise = sigma*(np.random.randn(row, col).astype(np.float32))
    img_noisy = data+noise
    img_noisy = np.clip(img_noisy, 0, 255)
    return img_noisy


def psnr_calc(img_orig, img_noise, out=False):
    err = (img_orig - img_noise)
    mse = np.mean(err**2)
    psnr = 20*math.log(img_orig.max()/np.sqrt(mse),10)
    if out==True:
        print(mse)
        print(psnr)
    return psnr
