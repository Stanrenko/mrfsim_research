set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Full MRF dictionary e.g. mrf_dict_optim/dico_TR2.49_reco2.99.pkl
# $3 : Nb segments (optional - default 1024)
# $4 : Example slice 1 (optional - default 180)
# $5 : Example slice 2 (optional - default 80)
# $6 : Index for parsing header file - default -1


INDEX_def=-1
NSEGMENTS_def=1024
SLICE1_def=180
SLICE2_def=80

GAMMA=0.7
GAMMASTR=$(echo "$GAMMA" | sed "s/\./"_"/")

NCOMP=12
NSING=6
NITER=0

NSEGMENTS=${3-${NSEGMENTS_def}}
SLICE1=${4-${SLICE1_def}}
SLICE2=${5-${SLICE2_def}}
INDEX=${6-${INDEX_def}}

SIMUS=1
US=1


#Extracting k-space and navigator data
echo "######################################################"
echo "Extracting k-space and navigator data"
python scripts/script_recoInVivo_3D_machines.py build_kdata --filename $1.dat --index ${INDEX} #--nb-rep 40 #--dens-adj False

# rm $1.npy
rm $1.dat


# Coil compression
echo "######################################################"
echo "Coil Compression $NCOMP virtual coils"
python scripts/script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP --spoke-start 400 --filename-seqParams $1_seqParams.pkl --lowmem True

# exit

rm $1_kdata.npy

# #Rebuild singular volumes for all bins
echo "######################################################"
echo "Rebuilding singular volumes"

python scripts/script_recoInVivo_3D_machines.py build_volumes_singular --filename-kdata $1_bart${NCOMP}_kdata.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --filename-seqParams $1_seqParams.pkl --L0 $NSING --dictfile $2 --useGPU False


python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative --filename-volume $1_bart${NCOMP}_volumes_singular.npy  --niter $NITER  --use-wavelet True --lambda-wav 1e-11 --mu 1 --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --filename-seqParams $1_seqParams.pkl #--filename-b1 data/InVivo/3D/patient.003.v19/meas_MID00088_FID70345_raFin_3D_tra_0_8x0_8x3mm_FULL_new_mrf_us2_b12Dplus1_12.npy



echo "######################################################"
echo "Building mask"
# python scripts/script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --l 0 --threshold 0.015 --it 2
python scripts/script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular_denoised.npy --l 0 --threshold 0.015 --it 1

# #Build maps for all bins
echo "######################################################"
echo "Building MRF maps"
#VOLUMESSUFFIX=ref0_it1
python scripts/script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart${NCOMP}_volumes_singular_denoised.npy" --filename-mask "$1_bart${NCOMP}_volumes_singular_denoised_l0_mask.npy" --dico-full-file $2 

python scripts/script_recoInVivo_3D_machines.py generate_image_maps --filename-map $1_bart${NCOMP}_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl --filename-seqParams $1_seqParams.pkl


# #Build maps for all bins
echo "######################################################"
echo "Reorienting maps and correcting for distortion"
filepath=$(dirname "$1.dat")
bash -i shellscripts/run_correct_maps.sh ${filepath}

# cp $1*.nii /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#6_2024_RespiMRF/2_Data_Raw/Spherical/dia.volunteer.011.v1