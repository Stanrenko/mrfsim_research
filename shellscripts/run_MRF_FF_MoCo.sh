python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp 16

python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -30 --top 30 --lambda-tv 0.00001 --incoherent False --ch 16 --ntimesteps 1

python script_recoInVivo_3D_machines.py build_volumes_allbins --filename-kdata $1_kdata.npy --ncomp 16
python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins --filename-volume $1 --filename-b1 $1_b12Dplus1_16.npy --mu-TV 0.5 --mu-bins 0.1 --gamma 0.4 --niter 5

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_4.npy --sl 46
cp $1_volumes_allbins_denoised_gamma_0_4.npy_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_4.npy --sl 20
cp $1_volumes_allbins_denoised_gamma_0_4.npy_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap

python  script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_volumes_allbins_denoised_gamma_0_4.npy

python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_volumes_allbins_denoised_gamma_0_4.npy --file-model $1_volumes_allbins_denoised_gamma_0_4_vxm_model_weights.h5

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_4_registered.npy --sl 46
cp $1_volumes_allbins_denoised_gamma_0_4_registered.npy_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_4_registered.npy --sl 20
cp $1_volumes_allbins_denoised_gamma_0_4_registered.npy_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap

python script_recoInVivo_3D_machines.py build_volumes_singular_allbins_registered --filename-kdata $1_kdata.npy --n-comp 16 --file-deformation-map 
$1_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --dictfile $2 --L0 10

python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular.npy --l 0 --threshold 0.015
cp $1_volumes_singular_allbins_registered_l0_mask.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap

python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_allbins_denoised_gamma_0_4_registered.npy --filename-mask $1_volumes_singular_allbins_registered_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular.json --filename $1.dat
