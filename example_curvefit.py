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
    
if __name__=="__main__":
  #create dummy data
  const=5
  tilt =0.1
  x = np.arange(-5,5, 0.5)
  y = gaussian_func(x, 10, 0, 1) + const*np.ones_like(x)+tilt*x+np.random.normal(0, 0.15, len(x))
  plt.scatter(x,y)

  init_params=[10,0,1,1,1]
  popt, pcov=fit_gaus_pol1(x,y,init_params)

  xd = np.arange(x.min(), x.max(), 0.01)
  y_fit=gaus_pol1(xd, popt[0], popt[1], popt[2], popt[3], popt[4])

  y_gaus=gaussian_func(xd, popt[0], popt[1], popt[2])
  y_pol1=xd*popt[4] + popt[3]

  plt.plot(xd, y_fit, color='r')
  plt.plot(xd, y_gaus, color='k', linestyle='--')
  plt.plot(xd, y_pol1, color='gray', linestyle='--')

  plt.show()

  
