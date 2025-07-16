set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $3 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict


NCOMP=46
NSING=6
NITER=0



# #Extracting k-space and navigator data
# echo "######################################################"
# echo "Extracting k-space and navigator data"
# python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

# rm $1.npy
# rm $1.dat

# #Coil compression
# echo "######################################################"
# echo "Coil Compression $NCOMP virtual coils"
# python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP --invert-dens-adj True --res 16
# # #python script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP
# # # exit
# # # #Rebuild singular volumes for all bins
# # # echo "######################################################"
# # echo "Rebuilding singular volumes"
# python script_recoInVivo_3D_machines.py build_volumes_singular --filename-kdata $1_kdata.npy --n-comp $NCOMP --L0 $NSING --dictfile $2 --useGPU False
# rm $1_kdata.npy

# # exit
# python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular.npy --l 0 --threshold 0.025 --it 1
# # python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart30_volumes_singular_gr0.npy --l 0 --threshold 0.015 --it 1
# # cp $1_bart30_volumes_singular_gr0_l0_mask.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# #Build maps for all bins
# #echo "######################################################"
# #echo "Building MRF maps for all bins"
# python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_volumes_singular.npy" --filename-mask "$1_volumes_singular_l0_mask.npy" --filename-b1 $1_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat

#python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart30_volumes_singular_gr0.npy" --filename-mask "$1_bart30_volumes_singular_gr0_l0_mask.npy" --filename-b1 $1_bart30_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat
#python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart${NCOMP}_volumes_singular.npy" --filename-mask "$1_bart${NCOMP}_volumes_singular_l0_mask.npy" --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat


#mkdir -p /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}

#cp $1_volumes_singular.npy /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}
#cp $1_volumes_singular_CF_iterative_2Dplus1_MRF_map_it0_*.mha /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}
#cp $1_volumes_singular_CF_iterative_2Dplus1_MRF_map.pkl /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}/

yes | sh run_MRF_mask_ROI_lowerbody_nodenoising.sh $1