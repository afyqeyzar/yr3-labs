import numpy as np
import matplotlib.pyplot as plt

def trig_func(h, p, L ):
  x_axis = np.arange(0, L + 0.01, 0.01)
  #print(x_axis)
  y_axis = []

  for i in x_axis:
    if i <= p:
      y_axis.append((h/p) * i)
    else:
      y_axis.append((h/(L-p))*(L-i))

  plt.plot(x_axis,y_axis,'black')
  #labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']
  #plt.xticks(x_axis, labels, rotation='vertical')  
  plt.title('Plot of initial conditions')
  plt.grid()
  plt.show()

  return x_axis

trig_func(0.03, 0.35, 0.7)