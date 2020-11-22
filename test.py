import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray 

import imgtoolkit
import waveshrink

# synthetic test, computed using lenna image and by adding awgn with choosen sigma

# set parameters
k = 1
sigma = 20
level = 4
shrink_type = 'bayes'
thresh_type = 'soft'

# load the image
img = Image.open('lenna.jpg')
img=np.asarray(img)

# add noise 
img_noisy = imgtoolkit.addnoise(img,sigma)

# compute psnr
imgtoolkit.psnr_calc(img,img_noisy,out=True)

# plot
plt.figure()
plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=255)

# image processing
img_denoised = waveshrink.shrink(img_noisy, level, shrink_type, thresh_type, k)

# compute psnr
imgtoolkit.psnr_calc(img,img_denoised,out=True)

# plot
plt.figure()
plt.imshow(img_denoised, cmap='gray', vmin=0, vmax=255)
plt.show()

pass