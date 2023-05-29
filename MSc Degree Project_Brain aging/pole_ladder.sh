#!/bin/bash

#-v: t2t /home/data/test_0420/registration_t2t/HC/T78toT75_0_vel.nii.gz
#-d: t2s /home/data/test_0420/registration_t2s/HC/T78toOAS30780_0_vel.nii.gz
#-m: mask /home/data/test_0420/mask/HC/OAS30780_age_78_0_mask.nii.gz
#-t: output /home/data/test_0420/registration_s2s/mha/HC/OAS30921_71to65_0_vel.mha

# Set up paths
disease_condn="AD/"
disease_code="_0_"
main_path="/home/data/test_0420/"
list_file="${main_path}models_and_data/test_subject_list_${disease_condn:0:-1}.txt"
v_t2t_path="${main_path}registration_t2t/${disease_condn}"
d_t2s_path="${main_path}registration_t2s/${disease_condn}"
m_mask_path="${main_path}mask/${disease_condn}"
t_s2s_path="${main_path}registration_s2s/mha/${disease_condn}"

# Read each line of the file
while IFS= read -r line; do
  # Remove square brackets and quotes
  line=${line//[\'\[\]]/}

  # Split line into an array
  IFS=', ' read -ra arr <<< "$line"

  arr[4]=$(echo "${arr[4]}" | grep -oE '[[:digit:]]+')

  # Construct paths and filenames
  v_t2t_file="${v_t2t_path}T${arr[2]}to${arr[4]}${disease_code}vel.nii.gz"
  d_t2s_file="${d_t2s_path}T${arr[2]}to${arr[0]}${disease_code}vel.nii.gz"
  m_mask_file="${m_mask_path}${arr[0]}_age_${arr[2]}${disease_code}mask.nii.gz"
  t_s2s_file="${t_s2s_path}${arr[0]}_${arr[2]}to${arr[4]}${disease_code}vel.mha"

  # Run command
  # /usr/src/myapp/Ladder/Ladder/build/SchildsLadder -v "$v_t2t_file" -d "$d_t2s_file" -m "$m_mask_file" -t "$t_s2s_file"
  /usr/src/myapp/Ladder/Ladder/build/SchildsLadder -v ${v_t2t_file} -d ${d_t2s_file} -m ${m_mask_file} -t ${t_s2s_file}
done < "$list_file"
