#!/bin/bash
source /home/fjr/anaconda3/etc/profile.d/conda.sh
conda activate SynthSeg

# python ./ext/SynthSeg/scripts/commands/SynthSeg_predict.py --i ${file_name} --o ${output} --robust

main_path='/home/fjr/data/trained_models/Atlas-GAN/my_plot_1e-4/corrected_nii/'
for file_name in $(ls ${main_path})
do
  if [ ${#file_name} -eq 22 ]
  then
    echo -e "${main_path}${file_name}"
    output=${file_name:0:-7}"_SynthSeg.nii.gz"
    echo -e "${output}"
    #echo -e "${#file_name}"
    python ./ext/SynthSeg/scripts/commands/SynthSeg_predict.py --i ${main_path}${file_name} --o ${main_path}${output} --robust
  fi
done