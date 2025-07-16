#set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Pre-scan data file for motion estimation e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi.dat
# $3 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $4 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict
# $5 : Nb segments (optional - default 1400)
# $6 : Example slice 1 (optional - default 46)
# $7 : Example slice 2 (optional - default 20)


NCOMP=16
NSING=6
NITER=0
NBINS=5

NSEGMENTS_def=1400
SLICE1_def=46
SLICE2_def=20

NSEGMENTS=${5-${NSEGMENTS_def}}
SLICE1=${6-${SLICE1_def}}
SLICE2=${7-${SLICE2_def}}


#Extracting k-space and navigator data
echo "######################################################"
echo "Extracting k-space and navigator data"
python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

#Coil compression
echo "######################################################"
echo "Coil Compression $NCOMP virtual coils"
python script_recoInVivo_3D_machines.py coil_compression --filename-kdata $1_kdata.npy --n-comp $NCOMP

echo "######################################################"
echo "Pre-scan navigator images $2_image_nav_diff.jpg"
echo "Based on step 1 navigator images, what is the best channel for motion estimation ?"
read CHANNEL
echo "Channel $CHANNEL will be used for motion estimation"

#Estimate displacement, bins and weights
echo "######################################################"
echo "Estimating displacement, bins and weights"
#python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -30 --top 30 --incoherent False --nb-segments ${NSEGMENTS} --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --seasonal-adj True --hard-interp True --nbins $NBINS --retained-categories "1,2,3"

NBINS=3

cp $1_displacement.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo
echo "Please check displacement $1_displacement.jpg. Press any key to continue..."
read DUMMY

#Rebuild singular volumes for all bins
echo "######################################################"
echo "Rebuilding singular volumes for all bins"
python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata $1_kdata.npy --n-comp $NCOMP --L0 $NSING --dictfile $3 --gating-only True

#Denoise singular volumes for all bins by using the motion field
echo "######################################################"
echo "Denoising singular volumes for all bins by using the motion field $2_volumes_allbins_registered_allindex_deformation_map.npy"
python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_volumes_singular_allbins.npy --filename-b1 $1_b12Dplus1_$NCOMP.npy --niter $NITER --file-deformation $2_volumes_allbins_registered_allindex_deformation_map.npy --filename-weights $1_weights.npy --mu-TV 0.1 --gating-only True


#Extract bins

for i in `seq 0 $(($NBINS-1))`
do
    python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr $i
done

#Build masks for all bins
for i in `seq 0 $(($NBINS-1))`
do
    python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr$i.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2
done

#Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all bins"
for i in `seq 0 $(($NBINS-1))`
do
    #echo $1_volumes_singular_gr$i\_l0_mask.npy
    python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_volumes_singular_gr$i.npy_volumes_allbins_registered_allindex.npy" --filename-mask "$1_volumes_singular_gr${i}_l0_mask.npy" --filename-b1 $1_b12Dplus1_${NCOMP}.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --filename $1.dat
done

#Extract gif for FF and water T1 for two example slices
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_ff.mha --sl ${SLICE1} --nb-gr $NBINS
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_wT1.mha --sl ${SLICE1} --nb-gr $NBINS
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_ff.mha --sl ${SLICE2} --nb-gr $NBINS
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_wT1.mha --sl ${SLICE2} --nb-gr $NBINS

mkdir -p /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}

cp $1_volumes_singular_gr*_CF_iterative_2Dplus1_MRF_map_it0_ff.mha /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}
cp $1_volumes_singular_gr*_CF_iterative_2Dplus1_MRF_map_it0_wT1.mha /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}


cp $1_volumes_singular_gr$(($NBINS))_CF_iterative_2Dplus1_MRF_map_it0_ff_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}
cp $1_volumes_singular_gr$(($NBINS))_CF_iterative_2Dplus1_MRF_map_it0_wT1_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}
cp $1_volumes_singular_gr$(($NBINS))_CF_iterative_2Dplus1_MRF_map_it0_ff_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}
cp $1_volumes_singular_gr$(($NBINS))_CF_iterative_2Dplus1_MRF_map_it0_wT1_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}

cp $1_volumes_singular_gr*.pkl /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo/comp${NCOMP}L0${NSING}BINS${NBINS}NITER${NITER}
