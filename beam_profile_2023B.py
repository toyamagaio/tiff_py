# coding: utf-8
import tifffile as tf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def plot_xypjt(imagefile, bg_imagefile, figfile, threshold, max_threshold, xmin, xmax, ymin, ymax):

  image   =tf.imread(imagefile)
  bg_image=tf.imread(bg_imagefile)
  image_sub= image-bg_image
  
  binary_image=(image>threshold)*255
  tf.imshow(binary_image,cmap='gray')
  #tf.imshow(image,cmap='gray')

  image_cut=image[ymin:ymax, xmin:xmax]
  bg_image_cut=bg_image[ymin:ymax, xmin:xmax]
  binary_image_cut=(image_cut>threshold)*255
  tf.imshow(binary_image_cut,cmap='gray')
  
  image_cut_mask=(image_cut>max_threshold)*0
  
  
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
    print(r,max_x,max_idx)
  
  pix2mm=90./(lm_idx[1]-lm_idx[0])
  
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
  plot_xypjt(imagefile, bg_imagefile, figfile, threshold, max_threshold, xmin, xmax, ymin, ymax)
