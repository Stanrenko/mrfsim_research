#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat
# $2 : Nb segments (optional - default 1400)
# $3 : Example slice 1 (optional - default 46)
# $4 : Example slice 2 (optional - default 20)

 
DICTFILE="mrf_dictconf_Dico2_Invivo_overshoot_B1df_adjusted_2.16_reco4.0_w8_simmean.dict"
DICTFILE_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_adjusted_2.16_reco4.0_w8_simmean.dict"

DICTFILE_LOWBODY="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"
DICTFILE_LOWBODY_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"

targetfolder="data/InVivo/3D/processing"


NSEGMENTS_def=1400
SLICE1_def=46
SLICE2_def=20

NSEGMENTS=${2-${NSEGMENTS_def}}
SLICE1=${3-${SLICE1_def}}
SLICE2=${4-${SLICE2_def}}


line=$(find $1 -type f -name "*respi*.dat")
echo $line
filename=$(basename "$line")
filepath=$(dirname "$line")
file_respi="${filename%.*}"
file_respi_full=${targetfolder}/${file_respi}

echo "######################################################"
echo "Processing motion scan data"



# echo ${filepath}/${file_respi}
# echo "Copying ${filepath}/${filename} in ${targetfolder}"
# cp ${filepath}/${filename} ${targetfolder}
# echo "Copied ${filepath}/${filename} in ${targetfolder}"



# echo ${file_respi_full}
# ##### Extracting k-space and navigator data
# echo "######################################################"
# echo "Extracting k-space and navigator data"
# python script_recoInVivo_3D_machines.py build_kdata --filename ${file_respi_full}.dat #--select-first-rep True --suffix "_firstrep"

# rm ${file_respi_full}.dat
# rm ${file_respi_full}.npy

# echo "Building navigator images to help with channel choice"
# python script_recoInVivo_3D_machines.py build_navigator_images --filename-nav-save ${file_respi_full}_nav.npy
# cp ${file_respi_full}_image_nav.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# cp ${file_respi_full}_image_nav_diff.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# echo "######################################################"
# echo "Based on the navigator images, what is the channel with best contrast for motion estimation ?"
# read CHANNEL
# echo "Channel $CHANNEL will be used for motion estimation"

sh run_MRF_MoCo_step1_us_bart_global_model_nochannel.sh ${file_respi_full} ${CHANNEL} ${NSEGMENTS} ${SLICE1} ${SLICE2}
exit
# read -p "Press enter to continue"

EXAM=-4
find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do

    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    # if [ $EXAM -eq -4 ]
    # then
    #     EXAM=$((EXAM+1))
    #     continue
    # fi


    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"

    file_mrf_full=${targetfolder}/${file_mrf}

    if [ $EXAM -eq -4 ]
    then
        echo "Processing respi mrf"
        sh run_MRF_MoCo_step2_onebin_bart_nochannel.sh ${file_mrf_full} ${file_respi_full} ${DICTFILE} ${DICTFILE_LIGHT} ${CHANNEL} ${NSEGMENTS} ${SLICE1} ${SLICE2}
        
    else
        echo "Processing lower body mrf"  
        
        sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
        echo "##############################################"
        if [ $EXAM -eq -2 ]
        then
            echo "Thighs: Automatic segmentation and ROI values"
            # sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
            yes | sh run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-thighs:model3
            yes | sh run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-thighs:model3
        fi

        if [ $EXAM -eq -1 ]
        then
            echo "Legs: Automatic segmentation and ROI values"
            # sh run_MRF_3D_denoised.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${DICTFILE_LOWBODY_LIGHT} ${EXAM}
            yes | sh run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-legs:model1
            yes | sh run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-legs:model1
        fi

    fi

    EXAM=$((EXAM+1))
done


# echo "######################################################"
# echo "Correcting in-phase and out-of-phase volumes for distortion"
# sh run_correct_volumes.sh ${targetfolder}

echo "######################################################"
echo "Generating whole-body maps"
echo "######################################################"
echo "Correcting maps for distortion and applying offset"
sh run_correct_maps.sh ${targetfolder}

echo "######################################################"
echo "Creating whole-body FF and T1H2O maps"
python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "wT1" --overlap 0.0,0.0,50.0
python script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ff" --overlap 0.0,0.0,50.0


echo "######################################################"
echo "Moving results to ${1}"
# mv ${targetfolder}/* ${1}

