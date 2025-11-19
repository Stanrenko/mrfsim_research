#set -e

#Step 1 : Uses prescan acquistion to estimate the motion field
# $1 : Pre-scan data file for motion estimation e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi.dat
# $2 : Nb segments (optional - default 1400)
# $3 : Example slice 1 (optional - default 46)
# $4 : Example slice 2 (optional - default 20)

NCOMP=46
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
NSEGMENTS=${2-${NSEGMENTS_def}}
SLICE1=${3-${SLICE1_def}}
SLICE2=${4-${SLICE2_def}}



# # Extracting k-space and navigator data
# echo "######################################################"
# echo "Extracting k-space and navigator data"
# python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat --index -2 #--select-first-rep True --suffix "_firstrep"

# # rm $1.dat
# # rm $1.npy



# # # # ##
# # # # ### #Coil compression
# echo "######################################################"
# echo "Coil Compression $NCOMP virtual coils"
# # python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP --invert-dens-adj True --res 16 #--cc-res-kz 8 --cc-res 16 #--cc-res 256
# # cp $1_b12Dplus1_$NCOMP.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo




# # echo "######################################################"
# # echo "Rebuilding volume"
# # python script_recoInVivo_3D_machines.py build_volumes_allbins --filename-kdata $1_kdata.npy --n-comp $NCOMP --full-volume True

# # rm $1_kdata.npy

python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume $1_full_volume.npy --filedico $1_seqParams.pkl --nifti True

eval "$(conda shell.bash hook)"
conda activate distortion
gradient_unwarp.py ${1}_full_volume.nii ${1}_full_volume_corrected.nii siemens -g ../gradunwarp/coeff.grad --fovmin -0.4 --fovmax 0.4 -n
conda deactivate


python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume $1_full_volume_corrected.nii --filedico $1_seqParams.pkl --suffix "_offset" --apply-offset True --nifti True --reorient False