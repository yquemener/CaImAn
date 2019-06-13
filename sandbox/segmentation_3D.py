#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:15:56 2018

@author: agiovann
"""


import numpy as np
import matplotlib.pyplot as pl
from time import time
import caiman as cm
import cv2
import scipy
from skimage import morphology
import itertools
#%%
#dat_ = cm.load('/home/andrea/Dropbox/NEL/DendriticData/495D2G_MC_ds_4_400_400_sparse_-2.hdf5')
dat_ = cm.load('/home/andrea/Dropbox/NEL/DendriticData/495D2G_MC_ds_4_400_400_sparse_1.hdf5')

#%%
idx_max = np.argmax(dat_.max((1,2)))
dat = dat_[idx_max -20:idx_max +20]
dat = dat_[idx_max-2:idx_max+1].transpose([1,2,0]).squeeze()

pl.figure()
pl.imshow(dat,cmap='gray'); pl.colorbar()
#%%
min_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
new_mov = scipy.sparse.csc_matrix((dat_[0].size,1000),dtype=np.uint8)
As = []
new_mov = []

for count,idx_max in enumerate(range(1120,1220)):#range(dat_.shape[0]-2):
#    if idx_max%1 == 0:
    print(idx_max)
    dat = dat_[idx_max:idx_max+3].transpose([1,2,0]).squeeze()
    bilateral = cv2.bilateralFilter(dat,0,1,1)

#    bilateral  = restoration.denoise_bilateral(dat,sigma_spatial=0.1)

    sample = np.uint8((bilateral > 0.08)*255)
    if False:
        close_object_ = morphology.closing(sample, morphology.ball(3))
        close_object_ = scipy.ndimage.binary_closing(sample, structure=np.ones([3,3,3]))
    else:
        close_object_ = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel)

    close_object = close_object_[:,:,1]
#
#    if close_object.ndim == 3:
#        close_object = close_object.min(-1)
#
    if False:
        sure_bg = cv2.dilate(close_object,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(close_object,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(sample[:,:,:].astype(np.uint8),markers.astype(np.uint8))
        bilateral[markers == -1] = [255,0,0]
#        labels,num_features = scipy.ndimage.label(close_object,structure=None)
#        output = labels[:,:,1]
#        idx_good_neuron = np.where([np.sum(output==i)>min_size for i in np.unique(output)])[0][1:]
    else:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(close_object.astype(np.uint8), connectivity=4)
        idx_good_neuron = np.where(stats[:,-1]>min_size)[0][1:]

    dims = np.prod(dat_[0].shape)
    dims_t = dat_[0].shape
    A = np.zeros((dims ,len(idx_good_neuron)),dtype = np.uint8)
    new_img = np.zeros(dims_t,dtype=np.uint8)
    for comp_id,idxg in enumerate(idx_good_neuron):
        new_img[output==idxg] = np.uint8(comp_id)
        A[np.ravel_multi_index(np.where(output==idxg),dims_t),comp_id] = 1
        
    As.append(A)
    if count>1:
        union = np.reshape([np.maximum(a,b).sum() for a,b in itertools.product(As[count-1].T,As[count].T)],
                         [As[count-1].shape[-1],As[count].shape[-1]])
        intersect = As[0].T.dot(As[1])
        IOU = intersect/union
        
    new_mov.append(new_img)

    if cv2.waitKey(int(1)) & 0xFF == ord('q'):
        break
    else:
#        cv2.imshow('frame', (new_mov[:,count]/new_mov[:,count].max()).toarray().reshape(dims_t))
        cv2.imshow('frame',  A.mean(-1).reshape(dims_t))

cv2.destroyAllWindows()

#%%
min_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
new_mov = scipy.sparse.lil_matrix((dat_[0].size,1000),dtype=np.uint8)
n_frames = 5
As = []
new_mov = []
for count,idx_max in enumerate(range(0,1220)):#range(dat_.shape[0]-n_frames):
#    if idx_max%1 == 0:
    print(idx_max)
    dat = dat_[idx_max:idx_max+n_frames]
    xcorr = dat.local_correlations(eight_neighbours=True, swap_dim = False, order_mean=1)
    sample = np.uint8((xcorr > 0.5)*255)
    if False:
        close_object_ = morphology.closing(sample, morphology.ball(3))
        close_object_ = scipy.ndimage.binary_closing(sample, structure=np.ones([3,3,3]))
    else:
        close_object_ = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel)

    close_object = close_object_
#
#    if close_object.ndim == 3:
#        close_object = close_object.min(-1)
#
    if False:
        sure_bg = cv2.dilate(close_object,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(close_object,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(sample[:,:,:].astype(np.uint8),markers.astype(np.uint8))
        bilateral[markers == -1] = [255,0,0]
#        labels,num_features = scipy.ndimage.label(close_object,structure=None)
#        output = labels[:,:,1]
#        idx_good_neuron = np.where([np.sum(output==i)>min_size for i in np.unique(output)])[0][1:]
    else:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(close_object.astype(np.uint8), connectivity=4)
        idx_good_neuron = np.where(stats[:,-1]>min_size)[0][1:]

    dims = np.prod(dat_[0].shape)
    dims_t = dat_[0].shape
#    A = np.zeros((dims ,len(idx_good_neuron)),dtype = np.uint8)
    new_img = np.zeros(dims_t,dtype=np.uint8)
    for comp_id,idxg in enumerate(idx_good_neuron):
        new_img[output==idxg] = 255#np.uint8(comp_id)
#        A[output==idxg,comp_id] = 1
#    As.append(A)

    new_mov.append(new_img)

    if cv2.waitKey(int(1)) & 0xFF == ord('q'):
        break
    else:
#        cv2.imshow('frame', (new_mov[:,count]/new_mov[:,count].max()).toarray().reshape(dims_t))
        cv2.imshow('frame',  new_img)

cv2.destroyAllWindows()
#%%
from sklearn.metrics import jaccard_similarity_score

#%%
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #closing = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
    #ret, thresh = cv2.threshold(np.uint8(closing*255),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#    pl.imshow(close_object[:,:,:].mean(-1)>250, cmap='gray')


#%%
from scipy import stats
vmin, vmax = stats.scoreatpercentile(dat, (0.5, 99.5))
dat = np.clip(dat, vmin, vmax)
dat = (dat - vmin) / (vmax - vmin)
pl.figure()
pl.imshow(dat, cmap='gray', vmin=vmin,vmax=vmax); pl.colorbar()
#%%
from skimage import restoration
t1 = time()
bilateral = restoration.denoise_bilateral(dat,sigma_spatial=4.)
t2 = time()
print('time for bilateral filter: %f' %(t2 - t1))
pl.figure()
pl.imshow(bilateral,cmap='gray',vmax=.2); pl.colorbar()
sample = np.uint8((bilateral > 0.08)*255)
pl.imshow(sample, cmap='gray')

#%%
from skimage import exposure
hi_dat = exposure.histogram(dat)
hi_bilateral = exposure.histogram(bilateral)
pl.figure()
pl.plot(hi_dat[1], hi_dat[0], label='data')
pl.plot(hi_bilateral[1], hi_bilateral[0],
         label='bilateral')
pl.xlim(0, 0.5)
pl.legend()
pl.title('Histogram of voxel values')
#%%

#%%
from scipy import ndimage
sample = np.uint8(ndimage.binary_fill_holes(sample)*255)
pl.figure()
pl.imshow(sample, cmap='gray')
#%%
from scipy import ndimage as ndi
dist_transform = ndi.distance_transform_edt(sample[:,:,:], sampling=[1,1,1])

#dist_transform = cv2.distanceTransform(sample[:,:,1],cv2.DIST_L2,5)
pl.imshow(dist_transform[:,:,0])
#%%
#from skimage import morphology
#open_object = morphology.opening(sample, morphology.ball(3))
#pl.imshow(close_object, cmap='gray')
#%%
close_object = morphology.closing(sample, morphology.ball(1))
pl.imshow(close_object[:,:,:], cmap='gray')
#%%
bbox = ndimage.find_objects(close_object)
#%%
mask = close_object[bbox[0]]


roi_dat = dat[bbox[0]]
pl.imshow(roi_dat[10], cmap='gray')
pl.contour(mask[10])
#%%

t3 = time()
nlm = restoration.denoise_nl_means(dat[150:350,150:350],
                    patch_size=5, patch_distance=7,
                    h=0.12, multichannel=False)
t4 = time()
print('time for non-local means denoising: %f' %(t4 - t3))
pl.imshow(nlm, cmap='gray')
#%%

t5 = time()
tv = restoration.denoise_tv_chambolle(dat,
                                      weight=0.2)
t6 = time()
print('time for total variation: %f' %(t6 - t5))
pl.imshow(tv,cmap='gray')
#%%
hi_nlm = exposure.histogram(nlm)
hi_tv = exposure.histogram(tv)

pl.plot(hi_nlm[1], hi_nlm[0])
pl.plot(hi_tv[1], hi_tv[0])
pl.axvline(0.33, color='k', ls='--')
pl.axvline(0.4, color='k', ls='--')
#%%

markers = np.zeros(nlm.shape, dtype=np.uint8)
markers[nlm > 0.4] = 1
markers[nlm < 0.33] = 2

pl.imshow(markers, cmap='gray')
#pl.contour(markers[10], [0.5, 1.5])
#%%
from skimage import segmentation
rw = segmentation.random_walker(nlm, markers, beta=1000., mode='cg_mg')

pl.imshow(nlm[10], cmap='gray')
pl.contour(rw[10], [1.5])

clean_segmentation = morphology.remove_small_objects(rw == 1, 200)

from skimage import measure
labels = measure.label(clean_segmentation,
                        background=0)
print(labels.max())
labels = (labels + 1).astype(np.uint8)

