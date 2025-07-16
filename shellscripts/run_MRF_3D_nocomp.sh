set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Full MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict
# $3 : Light MRF dictionary e.g. mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict




#Extracting k-space and navigator data
echo "######################################################"
echo "Extracting k-space and navigator data"
#python script_recoInVivo_3D_machines.py build_kdata --filename $1.dat

#Coil compression
echo "######################################################"
echo "Coil Sensi"
#python script_recoInVivo_3D_machines.py build_coil_sensi --filename-kdata $1_kdata.npy


#Rebuild singular volumes for all bins
echo "######################################################"
echo "Rebuilding volumes"
#python script_recoInVivo_3D_machines.py build_volumes --filename-kdata $1_kdata.npy --use-GPU False

#python script_recoInVivo_3D_machines.py build_mask --filename-kdata $1_kdata.npy


#Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all bins"
python script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_volumes.npy" --filename-mask "$1_mask.npy" --filename-b1 $1_b1.npy --dictfile $2 --dictfile-light $3 --optimizer-config opt_config_twosteps.json --filename $1.dat

cp $1_volumes_CF_twosteps_MRF_map_it0_*.mha /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
cp $1_volumes_CF_twosteps_MRF_map.pkl /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo