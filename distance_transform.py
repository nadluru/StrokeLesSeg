import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
import shutil
from pathlib import Path
from settings import loader_settings
from scipy import ndimage

from nnUNet.nnunet.dataset_conversion.utils import generate_dataset_json


if __name__ == '__main__':
    base_path = 'atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/'
    ref_path = os.path.join(base_path, 'labelsTr_2')
    label_path = os.path.join(base_path, 'labelsTr')

    d = [0 for _ in range(3)]
    for path in sorted(os.listdir(ref_path)):
        file_path = os.path.join(ref_path, path)
        img = sitk.ReadImage(file_path)
        nda = sitk.GetArrayFromImage(img)
        base = copy.deepcopy(nda)

        distance = ndimage.distance_transform_edt(nda)
        base[(distance ** 2 > 4.5)] = 2

        d[0] += (base == 1).sum()
        d[1] += (base == 2).sum()
        d[2] += (base == 3).sum()
        print(d)

        base = base.astype(int)
        new_img = sitk.GetImageFromArray(base)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())
        sitk.WriteImage(new_img, os.path.join(label_path, path))

    print(d)

    generate_dataset_json(
        output_file=os.path.join(base_path, 'dataset.json'),
        imagesTr_dir=os.path.join(base_path, 'imagesTr'),
        imagesTs_dir=os.path.join(base_path, 'imagesTs'),
        modalities=('T1W',),
        labels={
            '0': 'background', 
            '1': 'd<=2', 
            '2': 'd>2'},
        dataset_name='ATLAS'
    )

    print("Done")