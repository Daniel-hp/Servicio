import numpy as np
import matplotlib.pyplot as plt
import glob, os
import sys

def max_subarray(A):
    max_ending_here = max_so_far = 0
    max_start = start = 0
    max_end = end = 0

    # the range is [max_start, max_end)
    for i, x in enumerate(A):
        if max_ending_here + x > 0:
            max_ending_here = max_ending_here + x
            end = i+1
        else:
            max_ending_here = 0
            start = end = i

        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            max_start = start
            max_end = end

    return (max_start, max_end)



"""
#nombres_directorios = [
                       "hielo", "hielov2", "hielov3", "hielov4", "hielov5", "hielov6",
                       "muerte", "muertev2","muertev6", "calaverav3", "calaverav4", "calaverav5",
                       "paz", "pazv2","pazv3", "pazv4", "pazv5", "pazv6",
                       "soldado", "soldadov2", "soldadov3", "soldadov4", "soldadov5", "soldadov5",
                       "sonar", "sonarv2", "sonarv3", "sonarv4", "sonarv5", "sonarv5",
                       "veinte", "veintev2", "veintev3", "veintev4", "veintev5", "veintev6"
                       ]
"""

n1 = ["veinte", "veintev2", "veintev3", "veintev4", "veintev5", "veintev6"]
n2 = ["hielo", "hielov2", "hielov3", "hielov4", "hielov5", "hielov6"]
n3 = ["muerte", "muertev2","muertev6", "calaverav3", "calaverav4", "calaverav5"]
n4 = ["paz", "pazv2","pazv3", "pazv4", "pazv5", "pazv6"]
n5 = ["soldado", "soldadov2", "soldadov3", "soldadov4", "soldadov5", "soldadov5"]
n6 = ["sonar", "sonarv2", "sonarv3", "sonarv4", "sonarv5", "sonarv5"]
ns = [n1] + [n2]  + [n3] + [n4] + [n5] + [n6]
for nombres_directorios in ns:
    for n, z in zip(nombres_directorios, range(len(nombres_directorios))):
        os.chdir(n)
        elementos = len(glob.glob("*.npy"))
        arr1_x = np.zeros(elementos)
        arr1_y = np.arange(0, elementos)
        arr2_x = np.zeros(elementos)
        arr2_y = np.arange(0, elementos)
        for file, x in zip(glob.glob("*.npy"), (range(elementos))):
            l = np.load(file)
            respuesta = [np.mean(x) for x in l]
            max_sec = max_subarray(respuesta)
            arr1_x[x] = np.mean(l[(max_sec[0]-1):max_sec[1]])
            respuesta = [np.std(x) for x in l]
            max_sec = max_subarray(respuesta)
            arr2_x[x] = np.std(l[max_sec[0]:max_sec[1]])


        #plt.figure()
        #plt.subplot(6, 6, z+1)
        plt.subplot(2, 3, z + 1)
        plt.errorbar(arr1_y, arr1_x, arr1_x/np.sqrt(elementos), linestyle='None', marker='^',label="1")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.errorbar(arr1_y, arr2_x, arr2_x/np.sqrt(elementos), linestyle='None', marker='^', label="2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Muestras")
        plt.ylabel("Valores")
        plt.title(n)
        plt.legend()
        #plt.savefig(str(n)+".png")

        os.chdir("../")
    plt.tight_layout()
    plt.savefig("v2" + str(nombres_directorios[0])+".png")
    plt.close()

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