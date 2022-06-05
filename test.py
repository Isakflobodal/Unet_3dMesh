import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

pts = np.array([[0,0], [0,1.1], [1,0], [1,1]])
tri = Delaunay(pts)

plt.triplot(pts[:,0], pts[:,1], tri.simplices)
plt.plot(pts[:,0], pts[:,1], 'o')
plt.show()

print(tri)
print(tri.simplices)