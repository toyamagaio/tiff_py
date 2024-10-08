# coding: utf-8
import tifffile as tf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import scipy
from scipy.optimize import curve_fit

def gaussian_func(x, amp, mu, sigma):
    return amp/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
def gaus_pol1(x, amp, mu, sigma, const, tilt):
    return gaussian_func(x,amp,mu,sigma)+const*np.ones_like(x)+tilt*x

def fit_gaus_pol1(x_data, y_data, init_params):
    popt, pcov = curve_fit(gaus_pol1, x_data, y_data, init_params)
    return popt, pcov

def plot_xypjt(imagefile, bg_imagefile, figfile, threshold, max_threshold, xmin, xmax, ymin, ymax):

  image   =tf.imread(imagefile)
  bg_image=tf.imread(bg_imagefile)
  image_sub= image-bg_image

  binary_bg_image=(bg_image>threshold)*255
  tf.imshow(binary_bg_image,cmap='gray')
  
  binary_image=(image>threshold)*255
  tf.imshow(binary_image,cmap='gray')
  #tf.imshow(image,cmap='gray')

  image_cut=image[ymin:ymax, xmin:xmax]
  bg_image_cut=bg_image[ymin:ymax, xmin:xmax]
  binary_image_cut=(image_cut>threshold)*255
  tf.imshow(binary_image_cut,cmap='gray')

  ymin1=100
  ymax1=200
  image_ycut=image[ymin1:ymax1, 100:-1]
  bg_image_ycut=bg_image[ymin1:ymax1, 100:-1]
  binary_image_ycut=(image_ycut>threshold)*255
  
  image_cut_mask=(image_cut>max_threshold)*0
  
  x_pjc_ycut   =np.sum(image_ycut   , axis=0)
  y_pjc_ycut   =np.sum(image_ycut   , axis=1)
  x_pjc_ycut_bg=np.sum(bg_image_ycut, axis=0)
  y_pjc_ycut_bg=np.sum(bg_image_ycut, axis=1)
  x_pjc_ycut_bgsub = x_pjc_ycut - x_pjc_ycut_bg
  y_pjc_ycut_bgsub = y_pjc_ycut - y_pjc_ycut_bg
  
  x_pjc = np.sum(image, axis=0)
  y_pjc = np.sum(image, axis=1)
  x_pjc_bg = np.sum(bg_image, axis=0)
  y_pjc_bg = np.sum(bg_image, axis=1)
  #x_pjc_bgsub = np.sum(image_sub, axis=0)
  #y_pjc_bgsub = np.sum(image_sub, axis=1)
  x_pjc_bgsub =x_pjc - x_pjc_bg
  y_pjc_bgsub =y_pjc - y_pjc_bg
  
  x_pjc_cut = np.sum(image_cut, axis=0)
  y_pjc_cut = np.sum(image_cut, axis=1)
  x_pjc_bg_cut = np.sum(bg_image_cut, axis=0)
  y_pjc_bg_cut = np.sum(bg_image_cut, axis=1)
  x_pjc_bgsub_cut =x_pjc_cut - x_pjc_bg_cut
  y_pjc_bgsub_cut =y_pjc_cut - y_pjc_bg_cut
  
  x_pix=np.arange(0,len(x_pjc),1)
  y_pix=np.arange(0,len(y_pjc),1)
  
  x_pix_ycut=np.arange(0,len(x_pjc_ycut),1)
  y_pix_ycut=np.arange(0,len(y_pjc_ycut),1)
  
  x_pix_cut=np.arange(0,len(x_pjc_cut),1)
  y_pix_cut=np.arange(0,len(y_pjc_cut),1)
  
  ##find local maximum (scintillator edge)
  ranges=[[100,200],[650,750]]
  lm_idx=list()
  lm_val=list()
  for r in ranges:
    max_x  =np.max(   x_pjc_bgsub[r[0]:r[1]])
    max_idx=np.argmax(x_pjc_bgsub[r[0]:r[1]])
    lm_val.append(max_x  )
    lm_idx.append(max_idx+r[0])
    print(r,max_x,max_idx+r[0])

  ##ycut
  print('ycut')
  lm_idx_ycut=list()
  lm_val_ycut=list()
  ranges_y=[[50,100],[550,650]]
  for r in ranges_y:
    max_x  =np.max(   x_pjc_ycut_bgsub[r[0]:r[1]])
    max_idx=np.argmax(x_pjc_ycut_bgsub[r[0]:r[1]])
    lm_val_ycut.append(max_x  )
    lm_idx_ycut.append(max_idx+r[0])
    print(r,max_x,max_idx+r[0])


  
  pix2mm=90./(lm_idx[1]-lm_idx[0])
  pix2mm_ycut=90./(lm_idx_ycut[1]-lm_idx_ycut[0])
  print('pix2mm=90./(lm_idx[1]-lm_idx[0])=90/({0}-{1})=90/{2}={3}'.format(lm_idx[1], lm_idx[0], (lm_idx[1]-lm_idx[0]), pix2mm))
  print('pix2mm_ycut=90./(lm_idx_ycut[1]-lm_idx_ycut[0])=90/({0}-{1})=90/{2}={3}'.format(lm_idx_ycut[1], lm_idx_ycut[0], (lm_idx_ycut[1]-lm_idx_ycut[0]), pix2mm_ycut))
  
  x_mm =x_pix*pix2mm
  y_mm =y_pix*pix2mm
  x_mm_cut =x_pix_cut*pix2mm
  y_mm_cut =y_pix_cut*pix2mm
  
  
  fig, ax=plt.subplots(2,2,figsize=[10,8])
  ax[0][0].plot(x_pjc)
  #ax[0][0].plot(x_pix,x_pjc,color='r')
  ax[1][0].plot(y_pjc)
  ax[0][0].plot(x_pjc_bg)
  ax[1][0].plot(y_pjc_bg)
  ax[0][1].plot(x_pjc_bgsub)
  ax[1][1].plot(y_pjc_bgsub)
  ax[0][1].plot(lm_idx,lm_val,marker='*',linestyle='',color='r')
  #ax[0].plot(x_pjc_cut)
  #ax[1].plot(y_pjc_cut)
  ax[0][0].set_title('x_projection')
  ax[1][0].set_title('y_projection')
  ax[0][1].set_title('x_pjc bg sub')
  ax[1][1].set_title('y_pjc bg sub')
  ax[0][0].set_xlabel('x pixel')
  ax[1][0].set_xlabel('y pixel')
  ax[0][1].set_xlabel('x pixel')
  ax[1][1].set_xlabel('y pixel')
  fig.tight_layout()
  
  fig1, ax1=plt.subplots(2,2,figsize=[10,8])
  ax1[0][0].plot(x_mm,x_pjc)
  ax1[1][0].plot(y_mm,y_pjc)
  ax1[0][0].plot(x_mm,x_pjc_bg)
  ax1[1][0].plot(y_mm,y_pjc_bg)
  ax1[0][1].plot(x_mm,x_pjc_bgsub,color='k')
  ax1[1][1].plot(y_mm,y_pjc_bgsub,color='k')
  ax1[0][0].set_title('x_projection')
  ax1[1][0].set_title('y_projection')
  ax1[0][1].set_title('x_pjc bg sub')
  ax1[1][1].set_title('y_pjc bg sub')
  ax1[0][0].set_xlabel('x [mm]')
  ax1[1][0].set_xlabel('y [mm]')
  ax1[0][1].set_xlabel('x [mm]')
  ax1[1][1].set_xlabel('y [mm]')
  fig1.tight_layout()
  
  fig2, ax2=plt.subplots(2,2,figsize=[10,8],sharex='all')
  ax2[0][0].plot(x_mm_cut,x_pjc_cut)
  ax2[1][0].plot(y_mm_cut,y_pjc_cut)
  ax2[0][0].plot(x_mm_cut,x_pjc_bg_cut)
  ax2[1][0].plot(y_mm_cut,y_pjc_bg_cut)
  ax2[0][1].plot(x_mm_cut,x_pjc_bgsub_cut,color='k')
  ax2[1][1].plot(y_mm_cut,y_pjc_bgsub_cut,color='k')
  ax2[0][0].set_title('x_projection (cut)')
  ax2[1][0].set_title('y_projection (cut)')
  ax2[0][1].set_title('x_pjc bg sub (cut)')
  ax2[1][1].set_title('y_pjc bg sub (cut)')
  ax2[0][0].set_xlabel('x [mm]')
  ax2[1][0].set_xlabel('y [mm]')
  ax2[0][1].set_xlabel('x [mm]')
  ax2[1][1].set_xlabel('y [mm]')
  fig2.tight_layout()

  tf.imshow(binary_image_ycut,cmap='gray')
  fig3, ax3=plt.subplots(2,2,figsize=[10,8],sharex='all')
  ax3[0][0].plot(x_pix_ycut,x_pjc_ycut)
  ax3[0][1].plot(y_pix_ycut,y_pjc_ycut)
  ax3[0][0].plot(x_pix_ycut,x_pjc_ycut_bg)
  ax3[0][1].plot(y_pix_ycut,y_pjc_ycut_bg)
  ax3[1][0].plot(x_pix_ycut,x_pjc_ycut_bgsub,color='k')
  ax3[1][1].plot(y_pix_ycut,y_pjc_ycut_bgsub,color='k')
  ax3[1][0].plot(lm_idx_ycut,lm_val_ycut,marker='*',linestyle='',color='r')
  ax3[0][0].set_title('x_projection (y cut)')
  ax3[0][1].set_title('y_projection (y cut)')
  ax3[1][0].set_title('x_pjc bg sub (y cut)')
  ax3[1][1].set_title('y_pjc bg sub (y cut)')
  ax3[0][0].set_xlabel('x [pix]')
  ax3[0][1].set_xlabel('y [pix]')
  ax3[1][0].set_xlabel('x [pix]')
  ax3[1][1].set_xlabel('y [pix]')
  fig3.tight_layout()

  fig4, ax4=plt.subplots(2,2,figsize=[10,8],sharex='all')

  init_params=[1e5,40,5,10,10]
  popt_h, pcov_h=fit_gaus_pol1(x_mm_cut,x_pjc_bgsub_cut,init_params)
  popt_v, pcov_v=fit_gaus_pol1(y_mm_cut,y_pjc_bgsub_cut,init_params)
  xd_mm_cut=np.arange(x_mm_cut.min(), x_mm_cut.max(), 0.1)
  yd_mm_cut=np.arange(y_mm_cut.min(), y_mm_cut.max(), 0.1)
  h_fit=gaus_pol1(xd_mm_cut, popt_h[0], popt_h[1], popt_h[2], popt_h[3], popt_h[4])
  v_fit=gaus_pol1(yd_mm_cut, popt_v[0], popt_v[1], popt_v[2], popt_v[3], popt_v[4])
  h_pol1=xd_mm_cut*popt_h[4]+popt_h[3]
  v_pol1=yd_mm_cut*popt_v[4]+popt_v[3]

  h_sigma=popt_h[2]
  v_sigma=popt_v[2]
  print('h_sigma: {0:.2f}, v_sigma: {1:.2f}'.format(h_sigma, v_sigma))

  ax4[0][0].plot(x_mm_cut,x_pjc_cut)
  ax4[1][0].plot(y_mm_cut,y_pjc_cut)
  ax4[0][0].plot(x_mm_cut,x_pjc_bg_cut)
  ax4[1][0].plot(y_mm_cut,y_pjc_bg_cut)
  ax4[0][1].plot(x_mm_cut,x_pjc_bgsub_cut,color='k')
  ax4[0][1].plot(xd_mm_cut,h_fit,color='red')
  ax4[0][1].plot(xd_mm_cut,h_pol1,color='gray',linestyle='--')
  ax4[1][1].plot(y_mm_cut,y_pjc_bgsub_cut,color='k')
  ax4[1][1].plot(yd_mm_cut,v_fit,color='red')
  ax4[1][1].plot(yd_mm_cut,v_pol1,color='gray',linestyle='--')
  ax4[0][0].set_title('x_projection (cut)')
  ax4[1][0].set_title('y_projection (cut)')
  ax4[0][1].set_title('x_pjc bg sub (cut)')
  ax4[1][1].set_title('y_pjc bg sub (cut)')
  ax4[0][0].set_xlabel('x [mm]')
  ax4[1][0].set_xlabel('y [mm]')
  ax4[0][1].set_xlabel('x [mm]')
  ax4[1][1].set_xlabel('y [mm]')
  fig4.tight_layout()
  
  pp=PdfPages(figfile)
  fignums = plt.get_fignums()
  print('fignums:',fignums)
  for fignum in fignums:
    print('--fignum:',fignum)
    plt.figure(fignum)
    pp.savefig()
  
  pp.close()
#plt.show()

##=============
if __name__=="__main__":
  imagefile   ='../20240205/#12_19.5MeV_11.5usecd_100nsecg_6minexp.tif'
  bg_imagefile='../20240205/#13_background_6minexp.tif'
  figfile='../fig/ccd_19.5MeV_20240205.pdf'
  xmin=200
  xmax=660
  ymin=80
  ymax=600
  threshold=1700
  max_threshold=1700
  print('x: {0} -- {1}'.format(xmin,xmax))
  print('y: {0} -- {1}'.format(ymin,ymax))
  plot_xypjt(imagefile, bg_imagefile, figfile, threshold, max_threshold, xmin, xmax, ymin, ymax)
