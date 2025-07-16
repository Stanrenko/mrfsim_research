#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat
# $2 : index of the exam to process (starting from -4)



# DICTFILE="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.1_reco2.99_w8_simmean.dict"
# DICTFILE_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.1_reco2.99_w8_simmean.dict"

DICTFILE_LOWBODY="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.06_reco2.99_w8_simmean.dict"
DICTFILE_LOWBODY_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.06_reco2.99_w8_simmean.dict"

targetfolder="data/InVivo/3D/processing"

INDEX=${2}


EXAM=-4
find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do

    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.}"

    if [ $EXAM -ne $INDEX ]
    
    then
        EXAM=$((EXAM+1))
        continue
    fi


    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"

    file_mrf_full=${targetfolder}/${file_mrf}

    sh shellscripts/run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
    

    EXAM=$((EXAM+1))
done




# echo "######################################################"
# echo "Generating whole-body maps"
# echo "######################################################"
# echo "Correcting maps for distortion and applying offset"
# sh shellscripts/run_correct_maps.sh ${targetfolder}

# echo "######################################################"
# echo "Creating whole-body FF and T1H2O maps"
# python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "wT1" --overlap 0.0,0.0,50.0
# python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ff" --overlap 0.0,0.0,50.0


# echo "######################################################"

# copyfolder=$(echo ${1} | sed 's/2_Data_Raw/3_Data_Processed/')
# echo "Moving results to ${copyfolder}"
# mv ${targetfolder}/* ${copyfolder}

