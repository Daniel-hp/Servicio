import numpy as np
import matplotlib.pyplot as plt

elementos = 85

magnitudes = list(range(0,elementos))
arr1_x=np.zeros(elementos)
arr1_y=np.zeros(elementos)
arr2_x=np.zeros(elementos)
arr2_y=np.zeros(elementos)
for x in range(elementos):
    l = np.load("sonarv2/magnitudes"+str(x)+".npy")
    arr1_x[x] = np.mean(l[:1])
    arr1_y[x] = np.mean(l[1:])
    arr2_x[x] = np.std(l[:1])
    arr2_y[x] = np.std(l[1:])


plt.errorbar(arr1_x, arr2_x, arr2_x/np.sqrt(elementos),linestyle='None', marker='^',label="X")
plt.errorbar(arr1_y, arr2_y, arr2_y/np.sqrt(elementos),linestyle='None', marker='^', label="Y")
plt.legend()
plt.savefig("Sonarv2.png")
"""
for x in magnitudes:
    arr = np.load("veinte/magnitudes"+str(x)+".npy")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.imshow(arr, interpolation='nearest', cmap=plt.cm.tab20c)
    plt.colorbar()
    plt.savefig("veinte/campo_vectorial"+str(x)+".png")
    plt.close()
"""