CHANNEL=3
NCOMP=16

#Extracting k-space and navigator data
python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

#Coil compression
python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP

#Estimate displacement, bins and weights
python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --use-ml True --incoherent False --ch $CHANNEL --ntimesteps 1 --nb-segments 1400 

#Rebuild and denoise volumes for all bins
python script_recoInVivo_3D_machines.py build_volumes_allbins --filename-kdata $1_kdata.npy --ncomp $NCOMP
python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins --filename-volume $1_volumes_allbins.npy --filename-b1 $1_b12Dplus1_$NCOMP.npy --mu-TV 0.5 --mu-bins 0. --gamma 0.8 --niter 5

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_8.npy --sl 46
cp $1_volumes_allbins_denoised_gamma_0_8.npy_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_8.npy --sl 20
cp $1_volumes_allbins_denoised_gamma_0_8.npy_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap


#Initialization of deformation fields
python  script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_volumes_allbins_denoised_gamma_0_8.npy
python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_volumes_allbins_denoised_gamma_0_8.npy --file-model $1_volumes_allbins_denoised_gamma_0_8_vxm_model_weights.h5

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_8_registered.npy --sl 46
cp $1_volumes_allbins_denoised_gamma_0_8_registered.npy_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_denoised_gamma_0_8_registered.npy --sl 20
cp $1_volumes_allbins_denoised_gamma_0_8_registered.npy_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap

#Refine volumes reconsturction with deformation field
python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_volumes_allbins.npy --filename-b1 $1_b12Dplus1_$NCOMP.npy --niter 1 --file-deformation $1_volumes_allbins_denoised_gamma_0_8_deformation_map.npy

#Refine deformation field estimation
python  script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_volumes_allbins_registered_allindex.npy
python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_volumes_allbins_registered_allindex.npy --file-model $1_volumes_allbins_registered_allindex_vxm_model_weights.h5

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_registered_allindex_registered.npy --sl 46
cp $1_volumes_allbins_registered_allindex_registered.npy_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_allbins_registered_allindex_registered.npy --sl 20
cp $1_volumes_allbins_registered_allindex_registered.npy_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap



