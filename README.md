# 3D MRF Whole-body - research repository


## Requirements

After cloning the repository, one should create the environment with python 3.9 and install dependencies by going at the root of the repo.
Cupy is required while keeping numpy version 1.21.0 due to conflicts with the tensorflow version used in voxelmorph.

```
conda create -n <env> python==3.9
conda activate <env>
pip install -e .
conda install -c conda-forge cupy=8.3.0 numpy=1.21.0
```


## Exemples
```
sh shellscripts/run_MRF_WholeBody_axial_new_dict.sh <path/to/.dat files>
```

## References
<a id="1">[1]</a> 

