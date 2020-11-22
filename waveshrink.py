import pywt
import pywt.data
import numpy as np
import statistics
import math
from scipy import optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import gennorm as gn

def shrink(img, lev=4, shrink_type='bayes', thresh_type='hard', k=1):

    # convert to float
    data = np.asarray(img)
    data = np.float32(data)
    
    # swt
    c = pywt.swt2(data, 'bior4.4', level = lev, trim_approx=True)
    
    # algorithm choosing
    if shrink_type=='visu':
        d = visushrink(c, lev, thresh_type, k)
    elif shrink_type=='sure':
        d = sureshrink(c, lev, thresh_type)
    elif shrink_type=='bayes':
        d = bayesshrink(c, lev, thresh_type, k)
    else:
        d = c

    img_denoised = pywt.iswt2(d, 'bior4.4')
    img_denoised = np.clip(img_denoised, 0, 255)
    return img_denoised


def visushrink(c, lev, thresh_type, k):
    d = []
    # number of pixels
    nrow = len(c[0])
    ncol = len(c[0][0])
    M = nrow*ncol
    t = math.sqrt(2*math.log(M))
    # noise MED estimate
    sigma_noise = np.median(abs(c[lev][2]))/0.6745
    print(sigma_noise)
    thresh = sigma_noise*t*k
    # base level processing
    A = c[0]
    d.append(A)
    # other levels processing
    for i in range(1, lev+1):
        H = threshold(c[i][0], thresh, thresh_type)
        V = threshold(c[i][1], thresh, thresh_type) 
        D = threshold(c[i][2], thresh, thresh_type)  
        d.append((H,V,D))
    return np.array(d,dtype=object)
    

def bayesshrink(c, lev, thresh_type, k):
    d = []
    # base level processing
    A = c[0]
    nrow = len(c[0])
    ncol = len(c[0][0])
    M = nrow*ncol
    # noise MED estimate
    sigma_noise2 = (np.median(abs(c[lev][2]))/0.6745)**2
    d.append(A)
    # other levels processing
    for i in range(1, lev+1):
        thresh = k*sigma_noise2/(np.sqrt(max((sum(sum(c[i][0]**2))/M-sigma_noise2),0.001)))
        H = threshold(c[i][0], thresh, thresh_type)
        thresh = k*sigma_noise2/(np.sqrt(max((sum(sum(c[i][1]**2))/M-sigma_noise2),0.001)))
        V = threshold(c[i][1], thresh, thresh_type) 
        thresh = k*sigma_noise2/(np.sqrt(max((sum(sum(c[i][2]**2))/M-sigma_noise2),0.001)))
        D = threshold(c[i][2], thresh, thresh_type)  
        d.append((H,V,D))
    return np.array(d,dtype=object)


def sureshrink(c, lev, thresh_type):
    d = []
    # base level processing
    A = c[0]
    # noise MED estimate
    sigma_noise = np.median(abs(c[lev][2]))/0.6745
    d.append(A)
    nrow = len(c[0])
    ncol = len(c[0][0])
    M = nrow*ncol
    # set max threshold level as the universal threshold
    sigma_noise = np.median(abs(c[lev][2]))/0.6745
    max_thresh = sigma_noise*math.sqrt(2*math.log(M,2))
    for i in range(1, lev+1):
        thresh = sure_threshold (c[i][0], M, sigma_noise, max_thresh)
        H = threshold(c[i][0], thresh, thresh_type)
        thresh = sure_threshold (c[i][1], M, sigma_noise, max_thresh)
        V = threshold(c[i][1], thresh, thresh_type) 
        thresh = sure_threshold (c[i][2], M, sigma_noise, max_thresh)
        D = threshold(c[i][2], thresh, thresh_type)  
        d.append((H,V,D))
        print(i)
    return np.array(d)


def sure_threshold (c, M, sigma_noise, tmax):
    n_iter = 10
    sigma_noise2 = (sigma_noise ** 2)
    c=c.ravel()
    t=np.linspace(0,tmax,n_iter)
    risk_best=sure(0, c, M, sigma_noise2)
    t_best=0
    for i in range(1,n_iter):
        risk=sure(t[i], c, M, sigma_noise2)
        if risk<risk_best:
            risk_best=risk
            t_best=t[i]
    return t_best


def sure(x, c, M, sigma_noise2):
    risk = sigma_noise2*M
    for a in c:
        risk = risk + min(a**2, x**2)
        if abs(a) <= x:
            risk = risk - 2*sigma_noise2
    return risk


def threshold(v, thresh, thresh_type):
    v=np.array(v)
    if thresh_type=='soft':
        v[abs(v)<thresh]=0
        v[v>thresh]-=thresh
        v[v<-thresh]+=thresh
    elif thresh_type=='hard':
        v[abs(v)<thresh]=0
    return v

def test_hp(img,lev):
    # function to test laplacian distribution of the sub bands
    data = np.asarray(img)
    # convert to float
    data = np.float32(data)
    c = pywt.swt2(data, 'bior4.4', level = lev, trim_approx=True)
    # convert matrix to array
    a=c[1][0].ravel()
    x=np.linspace(min(a),max(a),1000)
    # retrieve fitting parameters
    [u,b,c]=gn.fit(a)
    # compute pdf
    y=gn.pdf(x,u,b,c)
    # plot pdf
    plt.plot(x,y,color='r')
    # plot histogram 
    plt.hist(a,bins='auto',histtype='bar',density=True)
    plt.show()
    pass