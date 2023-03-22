import os
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['CMU']

#Mean squared error
x = np.array([1, 2, 3])
y = np.array([1, 2, 1])
n = 8
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)

torr_40  = pd.read_csv('20230201_1675243900876_40.csv')
torr_90  = pd.read_csv('20230201_1675245290038_90.csv')
torr_140 = pd.read_csv('20230201_1675246542849_140.csv')
torr_190 = pd.read_csv('20230201_1675247749171_190.csv')
torr_240 = pd.read_csv('20230201_1675248982420_240.csv')

def convert(dataframe):
    dataframe = dataframe.to_numpy()
    dataframe = dataframe.reshape(-1)
    return dataframe[::2], dataframe[1::2]

torr_40_time,  torr_40_volt  = convert(torr_40)
torr_90_time,  torr_90_volt  = convert(torr_90)
torr_140_time, torr_140_volt = convert(torr_140)
torr_190_time, torr_190_volt = convert(torr_190)
torr_240_time, torr_240_volt = convert(torr_240)

plot_1 = plt.figure(figsize=(12*0.9,10*0.9))
plt.grid(visible=True, linewidth=0.6)

plt.title('График зависимости $U(t)$', fontsize=14)
plt.xlim(xmin=0, xmax=1000)
plt.ylim(ymin=6, ymax=14)
plt.ylabel('Напряжение $U$, мВ', fontsize=12, rotation=90, ha='right')
plt.xlabel('Время $t$, с', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=16, size=10)
plt.grid(visible=True, linewidth=0.6)


plt.errorbar(
    torr_40_time,
    torr_40_volt,
    fmt='r.',
    linewidth=0,
    markersize=3,
    elinewidth=1,
    label='$P = 40$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_90_time,
    torr_90_volt,
    fmt='g.',
    linewidth=0,
    markersize=3,
    elinewidth=1,
    label='$P = 90$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_140_time,
    torr_140_volt,
    fmt='b.',
    linewidth=0,
    markersize=3,
    elinewidth=1,
    label='$P = 140$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_190_time,
    torr_190_volt,
    fmt='k.',
    linewidth=0,
    markersize=3,
    elinewidth=1,
    label='$P = 190$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_240_time,
    torr_240_volt,
    fmt='y.',
    linewidth=0,
    markersize=3,
    elinewidth=1,
    label='$P = 240$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)

plt.legend(fontsize = 14, markerscale = 5)
plt.show()

plot_1.savefig(
    'u(t).pdf',
    format='pdf',
    bbox_inches='tight',
    #pad_inches=4
)

##################################################################################################################################

plot_2 = plt.figure(figsize=(12*0.9,10*0.9))
plt.grid(visible=True, linewidth=0.6)

plt.title('График зависимости ln$U(t)$', fontsize=14)
plt.xlim(xmin=0, xmax=1000)
plt.ylim(ymin=1.8, ymax=2.8)
plt.ylabel(
    'ln $U$',
    fontsize=20,
    rotation=0,
    horizontalalignment='right'
)
plt.xlabel('$t$, с', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16, size=10)
plt.grid(visible=True, linewidth=0.6)



#yerr_1 = pressure_sub * 0.005
#xerr_1 = cons * 0.01

#z_1 = np.polyfit(torr_40_time, torr_40_volt, deg=1)
#polynom_1 = np.poly1d(z)
#plt.plot(, polynom_1(D),"r--", linewidth=1, label="МНК")

#z, err = np.polyfit(D, Q[1], deg=1, cov=True)
#polynom_2 = np.poly1d(z)
#plt.plot(D, polynom_2(D),"b--", linewidth=1, label="МНК")

plt.errorbar(
    torr_40_time,
    np.log(torr_40_volt),
    fmt='r.',
    linewidth=0,
    markersize=2,
    elinewidth=1,
    label='$P = 40$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_90_time,
    np.log(torr_90_volt),
    fmt='g.',
    linewidth=0,
    markersize=2,
    elinewidth=1,
    label='$P = 90$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_140_time,
    np.log(torr_140_volt),
    fmt='b.',
    linewidth=0,
    markersize=2,
    elinewidth=1,
    label='$P = 140$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_190_time,
    np.log(torr_190_volt),
    fmt='k.',
    linewidth=0,
    markersize=2,
    elinewidth=1,
    label='$P = 190$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)
plt.errorbar(
    torr_240_time,
    np.log(torr_240_volt),
    fmt='y.',
    linewidth=0,
    markersize=2,
    elinewidth=1,
    label='$P = 240$ торр',
    #xerr=xerr_1,
    #yerr=yerr_1
)


plt.legend(fontsize=12, markerscale = 5)
plt.show()

plot_2.savefig(
    'lnu(t).pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.5
)

##################################################################################################################################

#Mean squared error
x = torr_40_time
y = torr_40_volt
n = len(x)
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)
print('40 торр:', k, error_k, b)
print(- k * 5.3 * 775 / 2)

#Mean squared error
x = torr_90_time
y = torr_90_volt
n = len(x)
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)
print('90 торр:', k, error_k, b)
print(- k * 5.3 * 775 / 2)

#Mean squared error
x = torr_140_time
y = torr_140_volt
n = len(x)
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)
print('140 торр:', k, error_k, b)
print(- k * 5.3 * 775 / 2)

#Mean squared error
x = torr_190_time
y = torr_190_volt
n = len(x)
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)
print('190 торр:', k, error_k, b)
print(- k * 5.3 * 775 / 2)

#Mean squared error
x = torr_240_time
y = torr_240_volt
n = len(x)
k = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
b = np.mean(y) - k * np.mean(x)
k_o = np.mean(x*y) / np.mean(x ** 2)
error_k = 1 / np.sqrt(n) * np.sqrt(((np.mean(y * y)) - np.mean(y) ** 2) / (np.mean(x * x) - np.mean(x) ** 2) - k**2)
print('240 торр:', k, error_k, b)
print(- k * 5.3 * 775 / 2)

##################################################################################################################################

D = np.array([
    92.16511692575129,
    43.41401749079627,
    28.942937435850403,
    20.167484486521065,
    17.0915878236624    
])
p = np.linspace(40., 280., 5)

plot_3 = plt.figure(figsize=(12*0.9,10*0.9))
plt.grid(visible=True, linewidth=0.6)

plt.title('График зависимость $D(1/P)$', fontsize=18)
plt.xlim(xmin=0, xmax=0.03)
plt.ylim(ymin=0, ymax=100)
plt.ylabel('$D$, см$^2/$с', fontsize=20, rotation=0, ha='right')
plt.xlabel('$1/P$, торр', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16, size=10)
plt.grid(visible=True, linewidth=0.6)



yerr_1 = D[1] * 0.05
#xerr_1 =  * 0.01



#z, err = np.polyfit(D, Q[1], deg=1, cov=True)
#polynom_2 = np.poly1d(z)
#plt.plot(D, polynom_2(D),"b--", linewidth=1, label="МНК")

plt.errorbar(
    1/p,
    D,
    fmt='ks',
    linewidth=0,
    markersize=5,
    elinewidth=1,
    label='D(1/P)',
    #xerr=xerr_1,
    yerr=yerr_1,
    capsize=3
)

z = np.polyfit(1/p, D, deg=1)
polynom_1 = np.poly1d(z)
p = np.array([40, 100, 160, 220, 280, 760])
plt.plot(1/p, polynom_1(1/p),"r--", linewidth=1, label="МНК")


plt.legend(fontsize=18, markerscale = 1)
plt.show()

plot_3.savefig(
    'd(p).pdf',
    format='pdf',
    bbox_inches='tight',
    #pad_inches=4
)

##################################################################################################################################