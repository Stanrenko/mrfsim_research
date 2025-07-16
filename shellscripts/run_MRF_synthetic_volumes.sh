#set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf
# $2 : Model ()

eval "$(conda shell.bash hook)"
conda activate mrf


filename=$(basename "$1")
echo $filename

filepath=$(dirname "$1")
echo $filepath

currpath=$(pwd)
echo $currpath

# Extracting k-space and navigator data
echo "######################################################"
echo "Generating in-phase and out-of phase volumes"
# echo ${1}_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl
# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap ${1}_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl --reorient False
# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap ${1}_bart12_volumes_singular_allbins_volumes_allbins_registered_ref0_CF_iterative_2Dplus1_MRF_map.pkl --reorient False


find ${filepath} -type f -name "${filename}*MRF_map.pkl" | while read -r linemap ; do
    echo "Processing $linemap"
    python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap ${linemap} --reorient False

done



echo $filename

echo ${filepath}/${filename}
find ${filepath} -type f -name "${filename}*ip.mha" | while read -r line2 ; do
    echo "Processing $line2"
    #     # python script_recoInVivo_3D_machines.py getGeometry --filemha ${line2} --filename ${line} --suffix "_adjusted"
    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${line2} --filedico ${filepath}/${filename}_seqParams.pkl --nifti True

    filemha=$(basename "$line2")
    filemha_no_ext="${filemha%.*}"
    echo $filemha
    filenii=${filemha_no_ext}.nii
    echo $filenii
    filecorrected=${filemha_no_ext}_corrected.nii
        
    conda deactivate
    eval "$(conda shell.bash hook)"
    conda activate distortion
    gradient_unwarp.py ${filepath}/${filenii} ${filepath}/${filecorrected} siemens -g ../gradunwarp/coeff.grad --fovmin -0.2 --fovmax 0.2 -n
    conda deactivate

    eval "$(conda shell.bash hook)"
    conda activate mrf


    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filename}_seqParams.pkl --suffix "_offset" --apply-offset True --reorient False
    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filename}_seqParams.pkl --reorient False
    # python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filenii} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti False --reorient False
    
done
find ${filepath} -type f -name "${filename}*oop.mha" | while read -r line2 ; do
    echo "Processing $line2"
        # python script_recoInVivo_3D_machines.py getGeometry --filemha ${line2} --filename ${line} --suffix "_adjusted"
    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${line2} --filedico ${filepath}/${filename}_seqParams.pkl --nifti True

    filemha=$(basename "$line2")
    filemha_no_ext="${filemha%.*}"
    echo $filemha
    filenii=${filemha_no_ext}.nii
    echo $filenii
    filecorrected=${filemha_no_ext}_corrected.nii
        
    conda deactivate
    eval "$(conda shell.bash hook)"
    conda activate distortion
    gradient_unwarp.py ${filepath}/${filenii} ${filepath}/${filecorrected} siemens -g ../gradunwarp/coeff.grad --fovmin -0.2 --fovmax 0.2 -n
    conda deactivate

    eval "$(conda shell.bash hook)"
    conda activate mrf

    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filename}_seqParams.pkl --suffix "_offset" --apply-offset True --reorient False #--reorient True
    python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filename}_seqParams.pkl --reorient False
    # python script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filenii} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti False --reorient False
            
    
done

# python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $1_MRF_map.pkl

# exit 

# echo "######################################################"
# echo "Generating segmentation"
# eval "$(conda shell.bash hook)"
# conda activate mutools-dev-new
# museg-ai infer ${2} ${filename}_volumes_singular_denoised_ip_corrected_offset.mha ${filename}_volumes_singular_denoised_oop_corrected_offset.mha --root ${filepath} --filename ${filename}_volumes_singular_denoised_ip_corrected_offset.nii.gz --overwrite
# conda deactivate

