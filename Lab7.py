import os
print(os.getcwd())

import pandas as pd
import csv
file_path = "/Users/trevorgladstone/Desktop/ForceVsPositionLab7.csv"
df = pd.read_csv(file_path)
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
file = open(file_path, 'r')
print(file.read())
tx, Fdata, pos_data  = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,1,2), unpack=True)
f, axarr = plt.subplots(2, sharex=True)
plt.xlim(4, 35)
axarr[0].plot(tx, Fdata,'r')
axarr[0].set_title('0 degree Hookes Law')
axarr[0].set_ylabel('Force (N)')
axarr[1].plot(tx, pos_data)
axarr[1].set_xlabel('time (s)')
axarr[1].set_ylabel('Position (m)')
plt.show()
start_time = 4
stop_time = 35
in_range = (tx >= start_time) & (tx <= stop_time)
pos_sliced, F_sliced = pos_data[in_range], Fdata[in_range]
plt.scatter(pos_sliced, F_sliced)
plt.xlabel("Position (m)")
plt.ylabel("Force (N)")
plt.title ('ForceVsPosition')
plt.show()
def f_lin(x, m, c):
    return m*x + c
lin_opt, lin_cov = opt.curve_fit(f_lin, pos_sliced, F_sliced)
m, c = lin_opt
dm, dc = np.sqrt(np.diag(lin_cov))
print("m = %5.4f \u00b1 %5.4f" % (m, dm))
print("c = %5.4f \u00b1 %5.4f" % (c, dc))
F_pred = f_lin(pos_sliced, m, c)
fig1=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': '16'})
plt.scatter(pos_sliced, F_sliced)
plt.plot(pos_sliced, F_pred, color='orange')
plt.title("LSRL")
plt.xlabel("Position (m)")
plt.ylabel("Force (N)")
plt.show()
plt.scatter(pos_sliced, F_sliced-F_pred)
plt.axhline(y=0, color='orange')
plt.title("Residuals")
plt.xlabel("Position(m)")
plt.ylabel("Force residual (N)")
plt.show()