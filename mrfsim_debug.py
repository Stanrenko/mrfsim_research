from mrfsim import *
from machines import factory,MemoryStorage

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

storages = {"SearchMRF": MemoryStorage()}

with factory(hold=True, storages=storages):
    result=SearchMrf(path=r"data/KSpaceData.mat", dictfile="mrf175_CS.dict", metric="ls",niter=2, method="brute")
