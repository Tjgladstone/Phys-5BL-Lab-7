#Hooke's Law (0 degree case)
import os
print(os.getcwd())
import pandas as pd
import csv
file_path = "/Users/trevorgladstone/Desktop/try2.csv"
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
plt.xlim(12, 32)
axarr[0].plot(tx, Fdata,'r')
axarr[0].set_title('0 degree Hookes Law')
axarr[0].set_ylabel('Force (N)')
axarr[1].plot(tx, pos_data)
axarr[1].set_xlabel('time (s)')
axarr[1].set_ylabel('Position (m)')
plt.show()
start_time = 12
stop_time = 32
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

#Harmonic Oscillation (90 degree case)
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
file_path = "/Users/trevorgladstone/Desktop/60deg.csv"
file = open(file_path, 'r')
print(file.read())
tx, aydata, Fdata = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,2,4), unpack=True)
f, axarr = plt.subplots(2, sharex=True)
start_time = 10
stop_time = 32
tvals = np.arange(start_time, stop_time)
axarr[0].plot(tx, aydata,'r', label="accelerometer")
axarr[0].set_title('60 degree silver spring harmonic oscillator')
axarr[0].set_ylabel('Accelerometer (m/s2)')
axarr[1].plot(tx, Fdata)
axarr[1].set_xlabel('time (s)')
axarr[1].set_ylabel('Force (N)')
plt.xlabel("Time(s)")
plt.ylabel("Acceleration m/s^2")
plt.show()
yinterp = interp1d(tx, aydata, kind="linear")
start_time = 10
stop_time = 32
step_size = 0.001
tvals = np.arange(start_time, stop_time, step_size)  
yvals = yinterp(tvals)
plt.plot(tvals, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Harmonic Motion Data")
plt.show()
def simple_a(t, A, B, w, phi):
    return B - A * (w**2) * np.cos(w*t + phi)
oscillator_model = simple_a
Ainit, Binit, winit, phiinit, =[0.054, -9.8, 8.2, 1.1]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Simple Harmonic Motion Fit")
plt.show()
def vdamped_a(t, A, B, w, phi, beta):
    return B + A * np.exp(-beta*t) * ((beta**2-w**2)*np.cos(w*t + phi)+2*beta*w*np.sin(w*t+phi))
oscillator_model = vdamped_a
Ainit, Binit, winit, phiinit, betainit=[0.054, -9.8, 8.2, 1.1, 0.01205]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit, betainit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Damped Harmonic Motion Fit")
plt.show()
