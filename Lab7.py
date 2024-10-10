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
file_path = "/Users/trevorgladstone/Desktop/90deg.csv"
file = open(file_path, 'r')
tx, aydata, Fdata = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,2,4), unpack=True)
f, axarr = plt.subplots(2, sharex=True)
start_time = 10
stop_time = 32
tvals = np.arange(start_time, stop_time)
axarr[0].plot(tx, aydata,'r', label="accelerometer")
axarr[0].set_title('90 degree silver spring harmonic oscillator')
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
plt.title("Harmonic Motion Data 90°")
plt.show()
def simple_a(t, A, B, w, phi):
    return B - A * (w**2) * np.cos(w*t + phi)
oscillator_model = simple_a
Ainit, Binit, winit, phiinit, =[4.63975648e-02, -9.77158085e+00, 8.14193246e+00, 3.22244364e+00]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Simple Harmonic Motion Fit 90°")
plt.show()
start_pars=[Ainit, Binit, winit, phiinit]
pars, cov = opt.curve_fit(oscillator_model, tvals-start_time, yvals, p0=start_pars)
[A, B, w, phi] = pars
std_errs = np.sqrt(np.diag(cov))
print(np.transpose([pars, std_errs]))
ypred = oscillator_model(tvals-start_time, A, B, w, phi)
fig1=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': '16'})
plt.scatter(tvals, yvals)
plt.plot(tvals, ypred, color='orange')
plt.title("Simple Harmonic Motion Fit 90°")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (appropriate units)")
plt.show()
def vdamped_a(t, A, B, w, phi, beta):
    return B + A * np.exp(-beta*t) * ((beta**2-w**2)*np.cos(w*t + phi)+2*beta*w*np.sin(w*t+phi))
oscillator_model = vdamped_a
Ainit, Binit, winit, phiinit, betainit=[5.25438945e-02, -9.77170593e+00, 8.14222270e+00, 3.21731325e+00, 1.15502142e-02]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit, betainit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Damped Harmonic Motion Fit 90°")
plt.show()
start_pars=[Ainit, Binit, winit, phiinit, betainit]
pars, cov = opt.curve_fit(oscillator_model, tvals-start_time, yvals, p0=start_pars)
[A, B, w, phi, beta] = pars
std_errs = np.sqrt(np.diag(cov))
print(np.transpose([pars, std_errs]))
ypred = oscillator_model(tvals-start_time, A, B, w, phi, beta)
fig1=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': '16'})
plt.scatter(tvals, yvals)
plt.plot(tvals, ypred, color='orange')
plt.title("Damped Harmonic Motion Fit 90°")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (appropriate units)")
plt.show()

#Harmonic Oscillation (60 degree case)
file_path = "/Users/trevorgladstone/Desktop/60deg.csv"
file = open(file_path, 'r')
tx, aydata, Fdata = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,1,2), unpack=True)
f, axarr = plt.subplots(2, sharex=True)
start_time = 4.75
stop_time = 10.5
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
start_time = 4.75
stop_time = 10.5
step_size = 0.001
tvals = np.arange(start_time, stop_time, step_size)  
yvals = yinterp(tvals)
plt.plot(tvals, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Harmonic Motion Data 90°")
plt.show()
def simple_a(t, A, B, w, phi):
    return B - A * (w**2) * np.cos(w*t + phi)
oscillator_model = simple_a
Ainit, Binit, winit, phiinit, =[4.24954614e-02, -7.76388156e+00, 8.19655461e+00, 7.75128295e+00]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Simple Harmonic Motion Fit 60°")
plt.show()
start_pars=[Ainit, Binit, winit, phiinit]
pars, cov = opt.curve_fit(oscillator_model, tvals-start_time, yvals, p0=start_pars)
[A, B, w, phi] = pars
std_errs = np.sqrt(np.diag(cov))
print(np.transpose([pars, std_errs]))
ypred = oscillator_model(tvals-start_time, A, B, w, phi)
fig1=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': '16'})
plt.scatter(tvals, yvals)
plt.plot(tvals, ypred, color='orange')
plt.title("Simple Harmonic Motion Fit 60°")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (appropriate units)")
plt.show()
def vdamped_a(t, A, B, w, phi, beta):
    return B + A * np.exp(-beta*t) * ((beta**2-w**2)*np.cos(w*t + phi)+2*beta*w*np.sin(w*t+phi))
oscillator_model = vdamped_a
Ainit, Binit, winit, phiinit, betainit=[-9.00788245e-02, -7.79311098e+00, 8.18519423e+00, 4.53888720e+00, 2.97927399e-01]
plt.plot(tvals-start_time, oscillator_model(tvals-start_time, Ainit, Binit, winit, phiinit, betainit), color='orange')
plt.scatter(tvals-start_time, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Damped Harmonic Motion Fit 60°")
plt.show()
start_pars=[Ainit, Binit, winit, phiinit, betainit]
pars, cov = opt.curve_fit(oscillator_model, tvals-start_time, yvals, p0=start_pars)
[A, B, w, phi, beta] = pars
std_errs = np.sqrt(np.diag(cov))
print(np.transpose([pars, std_errs]))
ypred = oscillator_model(tvals-start_time, A, B, w, phi, beta)
fig1=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': '16'})
plt.scatter(tvals, yvals)
plt.plot(tvals, ypred, color='orange')
plt.title("Damped Harmonic Motion Fit 60°")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (appropriate units)")
plt.show()
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
file_path = "/Users/trevorgladstone/Desktop/60deg.csv"
file = open(file_path, 'r')
tx, aydata, Fdata = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,1,2), unpack=True)
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(tx, aydata,'r', label="accelerometer")
axarr[0].set_title('60 degree silver spring harmonic oscillator')
axarr[0].set_ylabel('Accelerometer (m/s2)')
axarr[1].plot(tx, Fdata)
axarr[1].set_xlabel('time (s)')
axarr[1].set_ylabel('Force (N)')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Harmonic Motion Data 60°")
plt.show()
yinterp = interp1d(tx, aydata, kind="linear")
start_time = 4.75
stop_time = 10.5
step_size = 0.015
tvals = np.arange(start_time, stop_time, step_size)  
yvals = yinterp(tvals)
plt.plot(tvals, yvals)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Harmonic Motion Data 60°")
plt.show()
from scipy.signal import argrelextrema as get_extrema_indices
neighborhood_size = 5
max_indices = get_extrema_indices(yvals, np.greater, order=neighborhood_size)
min_indices = get_extrema_indices(yvals, np.less, order=neighborhood_size)
print
print("Maxima found: ", np.size(max_indices))
print("Minima found: ", np.size(min_indices))
y_max = yvals[max_indices]
t_max = tvals[max_indices]
y_min = yvals[min_indices]
t_min = tvals[min_indices]
if len(t_max) > 1: 
    max_fit = np.polyfit(t_max, y_max, 1)  
    max_line = np.polyval(max_fit, tvals)  
if len(t_min) > 1:  
    min_fit = np.polyfit(t_min, y_min, 1)  
    min_line = np.polyval(min_fit, tvals)  
plt.scatter(tvals, yvals)
plt.scatter(t_max, y_max, color='orange')
plt.scatter(t_min, y_min, color='red')
if len(t_max) > 1:
    plt.plot(tvals, max_line, color='orange', linestyle='--', label='Maxima Fit')
    max_slope, max_intercept = max_fit
    print(f"Line of best fit for maxima: y = {max_slope:.4f}x + {max_intercept:.4f}")
if len(t_min) > 1:
    plt.plot(tvals, min_line, color='red', linestyle='--', label='Minima Fit')
    min_slope, min_intercept = min_fit
    print(f"Line of best fit for minima: y = {min_slope:.4f}x + {min_intercept:.4f}")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Maxima & Minima w/Line of Best")
plt.show()
