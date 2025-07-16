#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Pre-scan data file for motion estimation e.g. data/InVivo/3D/patient.003.v17/meas_MID00020_FID67000_raFin_3D_tra_1x1x5mm_FULL_new_respi.dat
# $3 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $4 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict

NCOMP=16
NSING=10
NITER=0

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
python script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -30 --top 30 --incoherent False --nb-segments 1400 --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --seasonal-adj True --hard-interp True

#Rebuild singular volumes for all bins
echo "######################################################"
echo "Rebuilding singular volumes for all bins"
python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata $1_kdata.npy --n-comp $NCOMP --L0 $NSING --dictfile $3 --gating-only True

#Denoise singular volumes for all bins by using the motion field
echo "######################################################"
echo "Denoising singular volumes for all bins by using the motion field $2_volumes_allbins_registered_allindex_deformation_map.npy"
python script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_volumes_singular_allbins.npy --filename-b1 $1_b12Dplus1_$NCOMP.npy --niter $NITER --file-deformation $2_volumes_allbins_registered_allindex_deformation_map.npy --filename-weights $1_weights.npy --mu-TV 0.5 --gating-only True


#Extract bins
python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr 0
python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr 1
python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr 2
python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr 3
python script_recoInVivo_3D_machines.py extract_allsingular_volumes_bin --file-volume $1_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --gr 4

#Build masks for all bins
python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr0.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2
python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr1.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2
python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr2.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2
python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr3.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2
python script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_volumes_singular_gr4.npy_volumes_allbins_registered_allindex.npy --l 0 --threshold 0.025 --it 2

#Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all bins"
python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_singular_gr0.npy_volumes_allbins_registered_allindex.npy --filename-mask $1_volumes_singular_gr0_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --nb-rep-center-part 10 --filename $1.dat
python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_singular_gr1.npy_volumes_allbins_registered_allindex.npy --filename-mask $1_volumes_singular_gr1_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --nb-rep-center-part 10 --filename $1.dat
python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_singular_gr2.npy_volumes_allbins_registered_allindex.npy --filename-mask $1_volumes_singular_gr2_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --nb-rep-center-part 10 --filename $1.dat
python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_singular_gr3.npy_volumes_allbins_registered_allindex.npy --filename-mask $1_volumes_singular_gr3_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --nb-rep-center-part 10 --filename $1.dat
python script_recoInVivo_3D_machines.py build_maps --filename-volume $1_volumes_singular_gr4.npy_volumes_allbins_registered_allindex.npy --filename-mask $1_volumes_singular_gr4_l0_mask.npy --filename-b1 $1_b12Dplus1_16.npy --dictfile $3 --dictfile-light $4 --optimizer-config opt_config_iterative_singular_shell.json --nb-rep-center-part 10 --filename $1.dat


#Extract gif for FF and water T1 for two example slices
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_ff.mha --sl 46
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_wT1.mha --sl 46
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_ff.mha --sl 20
python script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_volumes_singular_gr{}_CF_iterative_2Dplus1_MRF_map_it0_wT1.mha --sl 20

cp $1_volumes_singular_gr4_CF_iterative_2Dplus1_MRF_map_it0_ff_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo
cp $1_volumes_singular_gr4_CF_iterative_2Dplus1_MRF_map_it0_wT1_sl46_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo
cp $1_volumes_singular_gr4_CF_iterative_2Dplus1_MRF_map_it0_ff_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo
cp $1_volumes_singular_gr4_CF_iterative_2Dplus1_MRF_map_it0_wT1_sl20_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data\ Processed/log_MRF_MoCo

