#set -e

#Step 1 : Uses prescan acquistion to estimate the motion field
# $1 : Pre-scan data file for motion estimation e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi.dat
# $2 : Example slice 1 (optional - default 46)
# $3 : Example slice 2 (optional - default 20)

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

SLICE1=${3-${SLICE1_def}}
SLICE2=${4-${SLICE2_def}}


python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1.npy --file-model /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/test_global_model/multiple_patients_vxm_model_weights_torchaugment.h5 --config-train config_train_voxelmorph.json

python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_registered.npy --sl ${SLICE1}
cp $1_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_registered.npy --sl ${SLICE2}
cp $1_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
