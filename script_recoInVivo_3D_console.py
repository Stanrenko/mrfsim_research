
#import matplotlib
#matplotlib.use("TkAgg")


import sys, getopt

path = r"/home/cslioussarenko/PythonRepositories"
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv:m:d:s:p:")
    except getopt.GetoptError:
        print("script_reconinVivo_3D_console.py -v <volumes_file> -m <mask_file> -d <dict_file> -s <split> --p <pca_comp_number>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("script_reconinVivo_3D_console.py -v <volumes_file> -m <mask_file> -d <dict_file> -s <split> -p <pca_comp_number>")
            sys.exit()
        elif opt in ("-v"):
            filename_volume=arg
            volumes_all = np.load(arg)
        elif opt in ("-m"):
            mask = np.load(arg)
        elif opt in ("-d"):
            dictfile = arg
        elif opt in ("-s"):
            split=int(arg)
        elif opt in ("--pca"):
            threshold_pca = int(arg)

    output_file = filename_volume.split(".npy")[0] + "_MRF_map.pkl"

    optimizer = SimpleDictSearch(mask=mask,niter=0,seq=None,trajectory=None,split=split,pca=True,threshold_pca=threshold_pca,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other")
    all_maps=optimizer.search_patterns(dictfile,volumes_all)

    file = open(output_file, "wb")
    # dump information to that file
    pickle.dump(all_maps, file)
    # close the file
    file.close()

if __name__ == "__main__":
    main()