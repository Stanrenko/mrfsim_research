#set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf
# $2 : Model ()




filename=$(basename "$1")
echo $filename

filepath=$(dirname "$1")
echo $filepath

currpath=$(pwd)
echo $currpath



eval "$(conda shell.bash hook)"
conda activate mrf

#Extracting k-space and navigator data
echo "######################################################"
echo "Generating in-phase and out-of phase volumes"
python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $1_bart12_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl

# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $1_MRF_map.pkl

# exit 

eval "$(conda shell.bash hook)"
conda deactivate

echo "######################################################"de
echo "Generating segmentation"
eval "$(conda shell.bash hook)"
conda activate mutools-dev-new
museg-ai segment ${2} ${filename}_bart12_volumes_singular_denoised_ip.mha ${filename}_bart12_volumes_singular_denoised_oop.mha --root ${filepath} --filename ${filename}_bart12_volumes_singular_denoised_ip.nii.gz
conda deactivate

eval "$(conda shell.bash hook)"
conda activate mrf

# echo ${filename}_volumes_singular_ip.nii.gz
# mv ${currpath}/${filename}_volumes_singular_denoised_ip.nii.gz ${filepath}/${filename}_volumes_singular_denoised_ip.nii.gz

echo "######################################################"
echo "Generating ROI mask"
python script_recoInVivo_3D_machines.py generate_mask_roi_from_segmentation --fileseg $1_bart12_volumes_singular_denoised_ip.nii.gz

echo "######################################################"
echo "Calculating ROI results"
python script_recoInVivo_3D_machines.py getROIresults --filemap $1_bart12_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl --fileroi $1_bart12_volumes_singular_denoised_ip_maskROI.npy --filelabels ${filepath}/labels.txt --excluded 10 --kernel 5 --roi 100

