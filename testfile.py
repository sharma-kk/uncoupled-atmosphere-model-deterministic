import numpy as np
import matplotlib.pyplot as plt

y = np.linspace(0,1, 1001).reshape(-1,1)

# define piecewise defined funciton

h = lambda y: np.exp(1/((y - 0.25)*(y - 0.75)))/np.exp(-4/(0.75 - 0.25)**2)
g = lambda y: h(y) * np.where(y> 0.25, 1, 0)*np.where(y<.75, 1, 0)

# plot f(x)
plt.figure(1)
plt.clf()
plt.plot(y, g(y))
plt.title('f(x)')
plt.show(block=False)