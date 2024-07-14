import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
import shutil
from pathlib import Path
from settings import loader_settings

from nnUNet.nnunet.dataset_conversion.utils import generate_dataset_json


if __name__ == '__main__':
    # input_path = '/src/workspace/StrokeLesionSegmentation/data'  # Path for the input
    # output_path = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2'
    input_path = '/src/workspace/ATLAS3D'  # Path for the input
    output_path = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task103_ATLAS_v2_Self_Training'

    # if os.path.exists(os.path.join(output_path, 'imagesTr')):
    #     shutil.rmtree(os.path.join(output_path, 'imagesTr'))
    # os.mkdir(os.path.join(output_path, 'imagesTr'))

    # if os.path.exists(os.path.join(output_path, 'imagesTs')):
    #     shutil.rmtree(os.path.join(output_path, 'imagesTs'))
    # os.mkdir(os.path.join(output_path, 'imagesTs'))

    # if os.path.exists(os.path.join(output_path, 'labelsTr')):
    #     shutil.rmtree(os.path.join(output_path, 'labelsTr'))
    # os.mkdir(os.path.join(output_path, 'labelsTr'))

    input_path_1 = os.path.join(input_path, 'prediction_bids')
    file_name_list = os.listdir(input_path_1)  # List of files in the input
    file_path_list = [os.path.join(input_path_1, f, 'ses-1/anat') for f in file_name_list]

    for folder in file_path_list:
        if folder.endswith('.json/ses-1/anat'):
            continue
        for file_name in os.listdir(folder):
            fil = os.path.join(folder, file_name)
            # if 'T1w.nii.gz' in fil: # the suffix is .nii.gz
            #     out_name = os.path.basename(fil).replace('.nii.gz', '_0000.nii.gz')
            #     shutil.copyfile(fil, os.path.join(output_path, 'imagesTr', out_name))

            # elif 'label-L_desc-T1lesion_mask.nii.gz' in fil: # the suffix is .nii.gz
            if 'label-L_mask.nii.gz' in fil: # the suffix is .nii.gz
                out_name = os.path.basename(fil).replace('label-L_mask.nii.gz', 'T1w.nii.gz')
                img = sitk.ReadImage(fil)
                nda = sitk.GetArrayFromImage(img)
                if nda.max() < 1 - 1e-5:
                    img2 = sitk.BinaryThreshold(img, 1e-5, 1 - 1e-5, 1, 0 )
                    sitk.WriteImage(img2, os.path.join(output_path, 'labelsTr', out_name))
                else:
                    shutil.copyfile(fil, os.path.join(output_path, 'labelsTr', out_name))

        # else: # the suffix is not .nii.gz
        #     file_name = os.path.basename(fil)
        #     base_file_name = file_name.split('.')[0]
        #     # suffix_name = file_name.replace(base_file_name, '')
        #     file_sitk_img = sitk.ReadImage(fil)
        #     sitk.WriteImage(file_sitk_img, os.path.join(output_path, base_file_name + '_0000.nii.gz'))


    # input_path_2 = os.path.join(input_path, 'test/derivatives/ATLAS')
    # file_name_list = os.listdir(input_path_2)  # List of files in the input
    # file_path_list = [os.path.join(input_path_2, f, 'ses-1/anat') for f in file_name_list]

    # for folder in file_path_list:
    #     if folder.endswith('.json/ses-1/anat'):
    #         continue
    #     for file_name in os.listdir(folder):
    #         fil = os.path.join(folder, file_name)
    #         if 'T1w.nii.gz' in fil: # the suffix is .nii.gz
    #             out_name = os.path.basename(fil).replace('.nii.gz', '_0000.nii.gz')
    #             shutil.copyfile(fil, os.path.join(output_path, 'imagesTs', out_name))

    generate_dataset_json(
        output_file=os.path.join(output_path, 'dataset.json'),
        imagesTr_dir=os.path.join(output_path, 'imagesTr'),
        imagesTs_dir=os.path.join(output_path, 'imagesTs'),
        modalities=('T1W',),
        labels={'0': 'background', '1': 'lesion'},
        dataset_name='ATLAS'
    )

    print("Done")