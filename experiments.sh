# MRF-T1map experiments

## Dictionary
# dictionary with 8-spoke groups: `mrf175.dict`
# python mrfsim.py gendict --dictfile mrf175.dict 

## Experiments
# python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric ls --niter 0 --branch mrf175.ls.i0.brute 
# python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric nnls --niter 0 --branch mrf175.nnls.i0.brute

python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric ls --niter 1 --branch mrf175.nnls.i1.brute
python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method brute --metric nnls --niter 1 --branch mrf175.nnls.i1.brute

python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric ls --niter 0 --branch mrf175.ls.i0
python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric nnls --niter 0 --branch mrf175.nnls.i0

python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric ls --niter 1 --branch mrf175.ls.i1 
python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric nnls --niter 1 --branch mrf175.nnls.i1 

python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric ls --niter 2 --branch mrf175.ls.i2 
python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric nnls --niter 2 --branch mrf175.nnls.i2

python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric ls --niter 3 --branch mrf175.ls.i3 
python mrfsim.py search batch_phantoms.yml --dictfile  mrf175.dict --method group --metric nnls --niter 3 --branch mrf175.nnls.i3

