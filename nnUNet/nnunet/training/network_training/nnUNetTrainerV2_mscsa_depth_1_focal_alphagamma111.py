#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



import torch
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.network_architecture.generic_UNet_mscsa_depth_1 import Generic_UNet_MSCSA
from nnunet.training.network_training.nnUNetTrainerV2_mscsa_depth_1_SGD \
    import nnUNetTrainerV2_MSCSA_Depth_1_SGD
from nnunet.training.loss_functions.custom_loss import DL_and_Focal_loss


class nnUNetTrainerV2_MSCSA_Depth_1_Focal_AlphaGamma111(nnUNetTrainerV2_MSCSA_Depth_1_SGD):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        self.loss = DL_and_Focal_loss({'smooth': 1e-5}, {})
        self.loss.ce.gamma = torch.Tensor([1, 2, 1])
        self.loss.ce.aplha = [1, 0.5, 1]
