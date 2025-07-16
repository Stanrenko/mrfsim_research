#set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf




filename=$(basename "$1")
echo $filename

filepath=$(dirname "$1")
echo $filepath

currpath=$(pwd)
echo $currpath

#Extracting k-space and navigator data
# echo "######################################################"
# echo "Generating in-phase and out-of phase volumes"
# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $1_volumes_singular_CF_iterative_2Dplus1_MRF_map.pkl

# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $1_MRF_map.pkl



# exit 

echo "######################################################"
echo "Generating segmentation"
eval "$(conda shell.bash hook)"
conda activate mutools-dev-new
museg-ai infer museg-thighs:model3 ${filename}_volumes_singular_ip.mha ${filename}_volumes_singular_oop.mha --root ${filepath} --filename ${filename}_volumes_singular_ip.nii.gz

conda deactivate

echo ${filename}_volumes_singular_ip.nii.gz
# mv ${currpath}/${filename}_volumes_singular_ip.nii.gz

echo "######################################################"
echo "Generating ROI mask"
python script_recoInVivo_3D_machines.py generate_mask_roi_from_segmentation --fileseg $1_volumes_singular_ip.nii.gz

echo "######################################################"
echo "Calculating ROI results"
python script_recoInVivo_3D_machines.py getROIresults --filemap $1_volumes_singular_CF_iterative_2Dplus1_MRF_map.pkl --fileroi $1_volumes_singular_ip_maskROI.npy --filelabels ${filepath}/labels.txt

