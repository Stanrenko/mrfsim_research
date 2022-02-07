from mrfsim import *
from machines import factory,MemoryStorage

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#storages = {"SearchMRF": MemoryStorage()}
storages = {"GenDict": MemoryStorage()}


with factory(hold=True, storages=storages):
    #result=SearchMrf(path=r"data/KSpaceData.mat", dictfile="mrf175_CS.dict", metric="ls",niter=2, method="brute")
    GenDict(sequence_config="mrf_sequence.json", dict_config="mrf_dictconf_SimReco2.json", dictfile="mrf55_test.dict")