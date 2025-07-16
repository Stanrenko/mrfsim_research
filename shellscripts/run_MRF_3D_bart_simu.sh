set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $3 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict


NCOMP=15
NSING=6
NITER=0
NBINS=6
NKEPT=5


#Extracting k-space and navigator data
echo "######################################################"
echo "Extracting k-space and navigator data"
#python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

#Coil compression
# echo "######################################################"
echo "Coil Compression $NCOMP virtual coils"
python script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP #--filename-cc $2_bart_cc.cfl --calc-sensi False
#python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP



# echo "######################################################"
# echo "Generating random weights"
# python script_recoInVivo_3D_machines.py generate_random_weights --filename-kdata $1_kdata.npy --nbins $NBINS --nkept $NKEPT --equal-spoke-per-bin True

python script_recoInVivo_3D_machines.py build_volumes_singular --filename-kdata $1_bart${NCOMP}_kdata.npy --n-comp $NCOMP --L0 $NSING --dictfile $2 --useGPU False --filename-weights $1_weights.npy
#python script_recoInVivo_3D_machines.py build_volumes_singular --filename-kdata $1_kdata.npy --n-comp $NCOMP --L0 $NSING --dictfile $2 --useGPU False --filename-weights $1_weights.npy


python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular.npy --l 0 --threshold 0.015 --it 1
#python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular.npy --l 0 --threshold 0.015 --it 1


#Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all bins"
python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart${NCOMP}_volumes_singular.npy" --filename-mask "$1_bart${NCOMP}_volumes_singular_l0_mask.npy" --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat
#python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_volumes_singular.npy" --filename-mask "$1_volumes_singular_l0_mask.npy" --filename-b1 $1_b12Dplus1_${NCOMP}.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat

