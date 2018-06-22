import numpy as np
import matplotlib.pyplot as plt
import glob, os
nombres_directorios = [
                       "hielo", "hielov2", "hielov3", "hielov4", "hielov5", "hielov6",
                       "muerte", "muertev2","muertev6", "calaverav3", "calaverav4", "calaverav5",
                       "paz", "pazv2","pazv3", "pazv4", "pazv5", "pazv6",
                       "soldado", "soldadov2", "soldadov3", "soldadov4", "soldadov5", "soldadov5",
                       "sonar", "sonarv2", "sonarv3", "sonarv4", "sonarv5", "sonarv5",
                       "veinte", "veintev2", "veintev3", "veintev4", "veintev5", "veintev6"
                       ]








for n, z in zip(nombres_directorios, range(len(nombres_directorios))):
    os.chdir(n)
    elementos = len(glob.glob("*.npy"))
    arr1_x=np.zeros(elementos)
    arr1_y=np.zeros(elementos)
    arr2_x=np.zeros(elementos)
    arr2_y=np.zeros(elementos)
    for file, x in zip(glob.glob("*.npy"), (range(elementos))):
        l = np.load(file)
        arr1_x[x] = np.mean(l[:1])
        arr1_y[x] = np.mean(l[1:])
        arr2_x[x] = np.std(l[:1])
        arr2_y[x] = np.std(l[1:])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.subplot(6, 6, z+1)
    plt.errorbar(arr1_x, arr2_x, arr2_x/np.sqrt(elementos),linestyle='None', marker='^',label="X")
    plt.errorbar(arr1_y, arr2_y, arr2_y/np.sqrt(elementos),linestyle='None', marker='^', label="Y")
    plt.title(n)
    plt.legend()

    os.chdir("../")
plt.show()

"""
for x in range(elementos):
    arr = np.load("veintev2/magnitudes"+str(x)+".npy")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.imshow(arr, interpolation='nearest', cmap=plt.cm.tab20c)
    plt.colorbar()
    plt.savefig("veintev2/campo_vectorial"+str(x)+".png")
    plt.close()
"""