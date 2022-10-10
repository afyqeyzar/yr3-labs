import numpy as np
import matplotlib.pyplot as plt
import csv



time, voltage = np.loadtxt("./fft1.csv", skiprows = 1, delimiter = ',', unpack = True)

#print(time,voltage)

plt.plot(time,voltage)
plt.show()