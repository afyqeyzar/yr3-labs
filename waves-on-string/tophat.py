import numpy as np
import matplotlib.pyplot as plt

def tophat(start, end, end_of_graph, density):
  ini_plot_x = np.arange(0, end_of_graph, density)
  y = [0] * len(ini_plot_x)
  for idx, x in enumerate(ini_plot_x):
    if x >= start and x <= end:
      y[idx] = 1
    
  return ini_plot_x, y

# x, y = tophat(2, 4, 10, 0.1)


# print(x)
# print(y)
# plt.plot(x,y)
# plt.show()
