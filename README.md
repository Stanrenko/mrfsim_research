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

For coil compression, [bart](https://github.com/mrirecon/bart) is required - https://github.com/mrirecon/bart/blob/master/README 2.2 Downloading and Compilation. 



For correcting distortion due to large FOV, one should create a conda environment (name: distortion for shell scripts to work) and install [gradunwarp](https://github.com/Washington-University/gradunwarp) in it.

For automated segmentation, create a conda environment (name: mutools_dev_new for shell scripts to work) and follow instructions to install museg-ai: 
RMN_FILES\Outils\mutools\archives\mutools-dev\README.md

## Examples
Generating dico from .dat example
```
python scripts/script_recoInVivo_3D_machines.py mrf_gendict --datafile <path/to/.dat> --force True --dest mrf_dict_optim --wait-time 2.99 --seqfile dico/mrf_sequence_random_FA_varsp_varparam_allparams_2.22.json
```

Processing Whole-Body MRF
```
sh shellscripts/run_MRF_WholeBody_axial_new_dict.sh <path/to/.dat files>
```

Processing MoCo Kushball MRF (diaphragm)
```
sh shellscripts/run_MRF_MoCo_denoised_kushball_new_dict.sh <path/to/.dat files> <path/to/dico.pkl>
```

## References
<!-- <a id="1">[1]</a>  -->

