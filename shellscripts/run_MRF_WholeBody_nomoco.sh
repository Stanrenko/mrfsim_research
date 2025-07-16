#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat
# $2 : Nb segments (optional - default 1400)
# $3 : Example slice 1 (optional - default 46)
# $4 : Example slice 2 (optional - default 20)

 
DICTFILE="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.17_reco2.99_w8_simmean.dict"
DICTFILE_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.17_reco2.99_w8_simmean.dict"

DICTFILE_LOWBODY="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"
DICTFILE_LOWBODY_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"

targetfolder="data/InVivo/3D/processing"




# EXAM=-4
# find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do

#     echo $EXAM

#     filename=$(basename "$line")
#     filepath=$(dirname "$line")


#     file_mrf="${filename%.*}"

#     if [ $EXAM -ne -4 ]
#     then
#         EXAM=$((EXAM+1))
#         continue
#     fi


#     echo ${filepath}/${file_mrf}

    # echo "Copying ${filepath}/${filename} in ${targetfolder}"
    # cp ${filepath}/${filename} ${targetfolder}
    # echo "Copied ${filepath}/${filename} in ${targetfolder}"

#     file_mrf_full=${targetfolder}/${file_mrf}

#     if [ $EXAM -eq -4 ]
#     then
#         echo "Processing upper-body mrf"
#         sh shellscripts/run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE} ${DICTFILE_LIGHT} ${EXAM}        
#     else
#         echo "Processing lower body mrf"  
        
#         sh shellscripts/run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
#         echo "##############################################"
#         if [ $EXAM -eq -2 ]
#         then
#             echo "Thighs: Automatic segmentation and ROI values"
#             # sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
#             yes | sh shellscripts/run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-thighs:model3
#             yes | sh shellscripts/run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-thighs:model3 roi-thighs
#         fi

#         if [ $EXAM -eq -1 ]
#         then
#             echo "Legs: Automatic segmentation and ROI values"
#             # sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
#             yes | sh shellscripts/run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-legs:model1
#             yes | sh shellscripts/run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-legs:model1 roi-legs
#         fi

#     fi

#     EXAM=$((EXAM+1))
# done


# echo "######################################################"
# echo "Correcting in-phase and out-of-phase volumes for distortion"
# sh run_correct_volumes.sh ${targetfolder}

# echo "######################################################"
# echo "Generating whole-body maps"
# echo "######################################################"
# echo "Correcting maps for distortion and applying offset"
# sh shellscripts/run_correct_maps.sh ${targetfolder}

echo "######################################################"
echo "Creating whole-body FF and T1H2O maps"
python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "wT1" --overlap 0.0,0.0,50.0
python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ff" --overlap 0.0,0.0,50.0


# echo "######################################################"
# echo "Moving results to ${1}"
# # mv ${targetfolder}/* ${1}

