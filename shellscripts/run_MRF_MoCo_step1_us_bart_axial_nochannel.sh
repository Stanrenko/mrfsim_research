#set -e

#Step 1 : Uses prescan acquistion to estimate the motion field
# $1 : Pre-scan data file for motion estimation e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi.dat
# $2 : Channel for displacement calculation
# $3 : Nb segments (optional - default 1400)
# $4 : Example slice 1 (optional - default 46)
# $5 : Example slice 2 (optional - default 20)

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

echo $GAMMASTR
echo $WAVSTR
NSEGMENTS=${3-${NSEGMENTS_def}}
SLICE1=${4-${SLICE1_def}}
SLICE2=${5-${SLICE2_def}}
CHANNEL=${2}
#

SIMUS=1
US=1

# # ##
# # ### #Coil compression
echo "######################################################"
echo "Coil Compression $NCOMP virtual coils"

python scripts/script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP #--filename-cc $1_bart_cc.cfl
cp $1_bart${NCOMP}_b12Dplus1_${NCOMP}.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# rm $1_kdata.npy

# # ##
# # ##
# # #exit
# # ##
# # ##
# ## #Estimate displacement, bins and weights
echo "######################################################"
echo "Estimating displacement, bins and weights"
# # # # ###python scripts/script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --use-ml True --incoherent False --ch $CHANNEL --ntimesteps 1 --nb-segments $NSEGMENTS --equal-spoke-per-bin True --nbins $NBINS --retained-categories "0,1,2,3"
python scripts/script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -20 --top 45 --incoherent False --nb-segments ${NSEGMENTS} --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --nbins $NBINS --retained-categories "0,1,2,3,4" --sim-us $SIMUS --us $US --interp-bad-correl True --seasonal-adj False --pad 0 --nspoke-per-z 1 --soft-weight True #--stddisp 1.5
cp $1_displacement.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# # # # #exit


echo "######################################################"
echo "Rebuilding and denoising volumes for all bins"
python scripts/script_recoInVivo_3D_machines.py build_volumes_allbins --filename-kdata $1_bart${NCOMP}_kdata.npy --n-comp $NCOMP --filename-weights $1_weights.npy
# # ## ##
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_allbins.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_allbins.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
#exit
# ## #
# ## #
# ## #
# #exit
# exit
rm $1_bart${NCOMP}_kdata.npy

python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins --filename-volume $1_bart${NCOMP}_volumes_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --mu-TV 0.1 --mu-bins 0. --gamma $GAMMA --niter 5 --filename-weights $1_weights.npy --axis 1


python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_denoised_gamma_$GAMMASTR.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_allbins_denoised_gamma_$GAMMASTR.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_denoised_gamma_$GAMMASTR.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_allbins_denoised_gamma_$GAMMASTR.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# exit

#Initialization of deformation fields
echo "######################################################"
echo "Initialization of deformation field"
python scripts/script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}.npy --file-config-train config/config_train_voxelmorph_shell.json --nepochs 1500 --kept-bins "0,1,2,3" --axis 1 #--lr 0.0002 --init-weights $1_volumes_allbins_denoised_gamma_0_8_vxm_model_weights.h5
python scripts/script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}.npy --file-model $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_vxm_model_weights.h5 --file-config-train config/config_train_voxelmorph_shell.json --axis 1

python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_registered.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_registered.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


#Refine volumes reconstruction with deformation field
echo "#####################################################"
echo "Refining volumes reconstruction with deformation field"
python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_bart${NCOMP}_volumes_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --niter 1 --file-deformation $1_bart${NCOMP}_volumes_allbins_denoised_gamma_${GAMMASTR}_deformation_map.npy --filename-weights $1_weights.npy --axis 1 #--us 2

#python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_bart${NCOMP}_volumes_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --niter 1 --file-deformation $1_bart${NCOMP}_volumes_allbins_registered_allindex_deformation_map.npy --filename-weights $1_weights.npy #--us 2


python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

## exit


#Refine deformation field estimation
echo "######################################################"
echo "Refining deformation field estimation"
python  scripts/script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy --file-config-train config/config_train_voxelmorph_shell.json --nepochs 1500 --kept-bins "0,1,2,3" --axis 1

python scripts/script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volumes_allbins_registered_allindex.npy --file-model $1_bart${NCOMP}_volumes_allbins_registered_allindex_vxm_model_weights.h5 --file-config-train config/config_train_voxelmorph_shell.json --axis 1


python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered_allindex_registered.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_allbins_registered_allindex_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_allbins_registered_allindex_registered.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_allbins_registered_allindex_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


