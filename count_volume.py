import os
import SimpleITK as sitk
import numpy as np
import scipy
import scipy.ndimage


# labels = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/labelsTr'
labels = '/src/workspace/atlasv2/raw/nnUNet_raw_data/Task150_MS/labelsTr'

results = []
# for sample in os.listdir(labels):
#     target = sitk.ReadImage(os.path.join(labels, sample))
#     target = sitk.GetArrayFromImage(target)
#     target = (target != 0).astype(np.uint8)

#     total_volume = target.sum()

#     results.append([sample, total_volume])
for sample in os.listdir(labels):
    target = sitk.ReadImage(os.path.join(labels, sample))
    target = sitk.GetArrayFromImage(target)
    target = (target != 0).astype(np.uint8)

    labeled, num_lesions = scipy.ndimage.label(target.astype(bool))

    # flag = 0
    # total_volume = 0
    # for idx_lesion in range(1, num_lesions + 1):
    #     lesion_component = labeled == idx_lesion
    #     volume = lesion_component.sum()
    #     # total_volume += volume
    #     results.append([sample, idx_lesion, volume])
    #     print(sample, idx_lesion, volume)
    # results.append([sample, target.sum()])

    volumes = [0] * 3
    for idx_lesion in range(1, num_lesions + 1):
        lesion_component = labeled == idx_lesion
        volume = lesion_component.sum()
        
        if volume >= 1000:
            volumes[2] += volume
        elif volume >= 100:
            volumes[1] += volume
        else:
            volumes[0] += volume

    results.append([sample] + volumes)
    print(results[-1])


        # if volume > 1000:
        #     flag = 1
    
    # if not flag:
    #     # results.append([sample, total_volume])
    #     results.append([sample.split,])

with open('total_volume.txt', 'w') as f:
    for line in results:
        # print(line)
        print(','.join([str(i) for i in line]), file=f)
    # for line in results:
    #     print(line)
    #     print(line, file=f)

