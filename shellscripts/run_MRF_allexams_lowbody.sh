#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat

DICTFILE_LOWBODY="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"
DICTFILE_LOWBODY_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"

targetfolder="data/InVivo/3D/processing"

echo ${targetfolder}

find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do
    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"

    file_mrf_full=${targetfolder}/${file_mrf}

    
    echo "Processing lower body mrf"  
    sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT}
    

done




echo "######################################################"
echo "Moving results to ${1}"
mv ${targetfolder}/* ${1}

