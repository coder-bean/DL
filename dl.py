import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

#part b1:

x=np.linspace(-2*np.pi, 2*np.pi,1000)
y=np.sin(x)

    #splitting into 4 equal parts:
split_x=np.split(x,4)
split_y=np.split(y,4)

#part b2:

x_validation = np.random.uniform(-2*np.pi,2*np.pi,300)
#ANN TO GENERATE Y-VALS

#part b3:



