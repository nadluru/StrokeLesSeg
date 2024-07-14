import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
import shutil
from pathlib import Path
from settings import loader_settings
import scipy
import scipy.ndimage

from nnUNet.nnunet.dataset_conversion.utils import generate_dataset_json


if __name__ == '__main__':
    base_path = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/'
    ref_path = os.path.join(base_path, 'labelsTr_2')
    label_path = os.path.join(base_path, 'labelsTr')

    results = []
    for path in os.listdir(ref_path):

        file_path = os.path.join(ref_path, path)
        img = sitk.ReadImage(file_path)
        nda = sitk.GetArrayFromImage(img)
        base = np.zeros(nda.shape).astype(int)
        labeled, num_lesions = scipy.ndimage.label(nda.astype(bool))

        for idx_lesion in range(1, num_lesions + 1):
            lesion_component = labeled == idx_lesion
            volume = lesion_component.sum()

            if volume >= 10000:
                label = 4
            elif volume >= 1000:
                label = 3
            elif volume >= 100:
                label = 2
            else:
                label = 1

            base += lesion_component * label

        base = base.astype(int)
        new_img = sitk.GetImageFromArray(base)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())
        sitk.WriteImage(new_img, os.path.join(label_path, path))

        results.append([path] + [(base == i).sum() for i in range(1, round(base.max()) + 1)])
        print(results[-1])

    generate_dataset_json(
        output_file=os.path.join(base_path, 'dataset.json'),
        imagesTr_dir=os.path.join(base_path, 'imagesTr'),
        imagesTs_dir=os.path.join(base_path, 'imagesTs'),
        modalities=('T1W',),
        labels={'0': 'background', '1': 'tiny', '2': 'small', '3': 'medium', '4': 'large'},
        dataset_name='ATLAS'
    )

    with open('msl.txt', 'w') as f:
        for line in results:
            print(','.join([str(i) for i in line]), file=f)

    print("Done")