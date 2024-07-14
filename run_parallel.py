import os
import SimpleITK as sitk
from scoring import dice_coef, _lesion_f1_score, simple_lesion_count_difference, volume_difference, precision, sensitivity
import numpy as np
import torch
from multiprocessing import Pool

def get_results(predictions):
    print(predictions)
    samples = os.listdir(predictions)
    samples.sort()

    metric = {'dice': dice_coef,
                'f1': _lesion_f1_score,
                'count': simple_lesion_count_difference,
                'volume': volume_difference,
                'precision_voxel': precision,
                'recall_voxel': sensitivity,
                }
    f1_keys = (
        'f1_0', 'precision_0', 'recall_0',
        'f1_1', 'precision_1', 'recall_1',
        'f1_2', 'precision_2', 'recall_2',
        'f1_3', 'precision_3', 'recall_3',
        'f1_4', 'precision_4', 'recall_4',
    )
    result = {'dice': [],
                'count': [],
                'volume': [],
                'precision_voxel': [],
                'recall_voxel': [],
                }
    for key in f1_keys:
        result[key] = []
    
    with open(predictions + '.txt', 'w') as f:
        for sample in samples:
            if sample.endswith('.nii.gz'):
                data = sitk.ReadImage(os.path.join(labels, sample))
                data = sitk.GetArrayFromImage(data)
                target = sitk.ReadImage(os.path.join(predictions, sample))
                target = sitk.GetArrayFromImage(target)
                target = (target != 0).astype(np.uint8)

                for key in metric.keys():
                    # result[key] += metric[key](data, target, batchwise=True)
                    result_instance = metric[key](data, target)
                    if key == 'f1':
                        for idx, f1_key in enumerate(f1_keys):
                            result[f1_key].append(result_instance[idx])
                    else:
                        result[key].append(result_instance)

                l = [sample] + [result[key][-1] for key in result.keys()]

                print(','.join([str(i) for i in l]), file=f)
                print(','.join([str(i) for i in l]))

        for key in result.keys():
            l = torch.Tensor(result[key])
            std, mean = torch.std_mean(l)

            print(key, mean.item(), std.item(), file=f)
            print(key, mean.item(), std.item())


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default='original')
args = parser.parse_args()
    

labels = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/labelsTr'

folder_path = args.folder
predictions = []
for path in sorted(os.listdir(folder_path)):
    prediction_path = os.path.join(folder_path, path)

    if os.path.isdir(prediction_path):
        predictions.append(prediction_path)

threads = 10
p = Pool(threads)
p.starmap(get_results, zip(predictions))
p.close()
p.join()

print("Done")