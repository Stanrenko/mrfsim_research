#set -e

#Step 1 : Uses prescan acquistion to estimate the motion field
# $1 : MRF data file for motion estimation + mapping e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $3 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $4 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict
# $4 : Nb segments (optional - default 1400)
# $5 : Example slice 1 (optional - default 46)
# $6 : Example slice 2 (optional - default 20)

NCOMP=30
NBINS=5
NSEGMENTS_def=1400
SLICE1_def=46
SLICE2_def=20

#CURRPATH="$1"
#echo $CURRPATH
#
##FILENAME= echo $1 | rev | cut -d/ -f1 | rev
##FILEVAR=$($FILENAME)
#CURRFILE= eval printf "${CURRPATH##*/}"
#FILEVAR=$(echo "$CURRFILE")
#echo $FILEVAR
#
#echo "${FILEVAR}_volume_denoised_gr"

GAMMA=0.9
GAMMASTR=$(echo "$GAMMA" | sed "s/\./"_"/")
WAV=0.0000005
WAVSTR=$(echo "$WAV" | sed "s/\./"_"/")

NITER=5
NSING=6

echo $GAMMASTR
echo $WAVSTR
NSEGMENTS=${4-${NSEGMENTS_def}}
SLICE1=${5-${SLICE1_def}}
SLICE2=${6-${SLICE2_def}}

#
US=1
SIMUS=1

REF=0
##
#Extracting k-space and navigator data
# echo "######################################################"
# echo "Extracting k-space and navigator data"
# python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat #--select-first-rep True --suffix "_firstrep"

# rm $1.dat
# rm $1.npy

# echo "Building navigator images to help with channel choice"
# python script_recoInVivo_3D_machines.py build_navigator_images --filename-nav-save $1_nav.npy --seasonal-adj True
# cp $1_image_nav.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# cp $1_image_nav_diff.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# echo "######################################################"
# echo "Based on the navigator images, what is the channel with best contrast for motion estimation ?"
# read CHANNEL
# echo "Channel $CHANNEL will be used for motion estimation"

# # # ##
# # # ### #Coil compression
# echo "######################################################"
# echo "Coil Compression $NCOMP virtual coils"
# # # # #python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP --invert-dens-adj True --res 16 #--cc-res-kz 8 --cc-res 16 #--cc-res 256
# # # # #cp $1_b12Dplus1_$NCOMP.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# python script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP #--filename-cc $1_bart_cc.cfl
# cp $1_bart${NCOMP}_b12Dplus1_${NCOMP}.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# rm $1_kdata.npy

# # ##
# # ##
# # #exit
# # ##
# # ##
# # ## #Estimate displacement, bins and weights
# echo "######################################################"
# echo "Estimating displacement, bins and weights"
# # # ###python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --use-ml True --incoherent False --ch $CHANNEL --ntimesteps 1 --nb-segments $NSEGMENTS --equal-spoke-per-bin True --nbins $NBINS --retained-categories "0,1,2,3"
# python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -20 --top 45 --incoherent False --nb-segments ${NSEGMENTS} --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --nbins $NBINS --retained-categories "0,1,2,3,4" --sim-us $SIMUS --interp-bad-correl True --seasonal-adj True --randomize True --hard-interp True --pad 0 --nspoke-per-z 1 --soft-weight True #--stddisp 1.5
# cp $1_displacement.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# # # #exit

# # #Rebuild singular volumes for all bins
#echo "######################################################"
#echo "Rebuilding singular volumes for all bins"
#python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata $1_bart${NCOMP}_kdata.npy --L0 $NSING --dictfile $2 --useGPU False --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --filename-pca $1_bart${NCOMP}_virtualcoils_${NCOMP}.pkl --filename-weights $1_weights.npy --gating-only True
#exit

#echo "######################################################"
#echo "Extracting one singular volume for all bins"
#python script_recoInVivo_3D_machines.py extract_singular_volume_allbins --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy --l 0

# echo "######################################################"
# echo "Denoising singular volume for all bins"
# python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins --filename-volume $1_bart${NCOMP}_volume_singular_l0_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --mu-TV 0.1 --mu-bins 0. --gamma $GAMMA --niter 5 --filename-weights $1_weights.npy

# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy --sl ${SLICE1}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy --sl ${SLICE2}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo



# #Initialization of deformation fields
# echo "######################################################"
# echo "Initialization of deformation field"
# python script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}.npy --config-train config_train_voxelmorph_shell.json --nepochs 2000 --kept-bins "0,1,2,3" #--lr 0.0002 --init-weights $1_volumes_allbins_denoised_gamma_0_8_vxm_model_weights.h5
# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}.npy --file-model $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_vxm_model_weights.h5 --config-train config_train_voxelmorph_shell.json

# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy --sl ${SLICE1}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy --sl ${SLICE2}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# #Refine volumes reconstruction with deformation field
# echo "#####################################################"
# echo "Refining volumes reconstruction with deformation field"
#python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_bart${NCOMP}_volume_singular_l0_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --niter 1 --file-deformation $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_deformation_map.npy --filename-weights $1_weights.npy #--us 2


# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins.npy_volumes_allbins_registered_allindex.npy --sl ${SLICE1}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_registered_allindex.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_registered_allindex.npy --sl ${SLICE2}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_registered_allindex.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

#exit

#
#Refine deformation field estimation
echo "######################################################"
echo "Refining deformation field estimation"
#python  script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins.npy_volumes_allbins_registered_allindex.npy --config-train config_train_voxelmorph_shell.json --nepochs 1500 --kept-bins "0,1,2,3"

# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins.npy_volumes_allbins_registered_allindex.npy --file-model $1_bart${NCOMP}_volume_singular_l0_allbins_vxm_model_weights.h5 --config-train config_train_voxelmorph_shell.json


# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_registered.npy --sl ${SLICE1}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_registered.npy --sl ${SLICE2}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo



#python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volumes_allbins.npy --file-model ./data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi_firstrep_bart30_volumes_allbins_registered_allindex_vxm_model_weights.h5 --config-train config_train_voxelmorph_shell.json
#python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered.npy --sl ${SLICE1}
#cp $1_bart${NCOMP}_volumes_allbins_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
#python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered.npy --sl ${SLICE2}
#cp $1_bart${NCOMP}_volumes_allbins_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

NITER=2

if [ $((NITER)) -eq 0 ]
then
  VOLUMESSUFFIX=ref${REF}
else
  VOLUMESSUFFIX=ref${REF}_it$(($NITER-1))
fi
echo $VOLUMESSUFFIX

# echo "######################################################"
# echo "Denoising singular volumes for all bins by using the motion field $1_bart${NCOMP}_volume_singular_l0_allbins.npy_volumes_allbins_registered_allindex.npy"
# python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered --index-ref ${REF} --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy  --niter $NITER --file-deformation $1_bart${NCOMP}_volume_singular_l0_allbins_deformation_map.npy --filename-weights $1_weights.npy --use-wavelet True --lambda-wav 1e-5 --mu 0.1 --kept-bins "0,1,2,3,4" --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --gating-only True #--filename-b1 data/InVivo/3D/patient.003.v19/meas_MID00088_FID70345_raFin_3D_tra_0_8x0_8x3mm_FULL_new_mrf_us2_b12Dplus1_12.npy


python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --l 0 --threshold 0.015 --it 2
cp $1_bart${NCOMP}_volumes_singular_allbins_volumes_allbins_registered_${VOLUMESSUFFIX}_l0_mask.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# #Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all iter"
#VOLUMESSUFFIX=ref0_it1
python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy" --filename-mask "$1_bart${NCOMP}_volumes_singular_allbins_volumes_allbins_registered_${VOLUMESSUFFIX}_l0_mask.npy" --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat # --slices "80"
