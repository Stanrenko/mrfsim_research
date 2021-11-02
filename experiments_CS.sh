# MRF-T1map experiments

## Dictionary
# dictionary with 8-spoke groups: `mrf175.dict`
# python mrfsim.py gendict --dictfile mrf175.dict 

## Experiments
# python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric ls --niter 0 --branch mrf175.ls.i0.brute 
# python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric nnls --niter 0 --branch mrf175.nnls.i0.brute

python mrfsim.py search batch_test.yml --dictfile  mrf175.dict --method brute --metric nnls --niter 0 --branch mrf175.nnls.i0.brute

python mrfsim.py gendict --dictfile mrf175_SimReco2_window_1.dict --dict-config mrf_dictconf_SimReco2.json

python mrfsim.py gendict --dictfile mrf175_Dico2_Invivo.dict --dict-config mrf_dictconf_Dico2_Invivo.json