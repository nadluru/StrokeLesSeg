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
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNet_variants.data_augmentation.nnUNetTrainerV2_DA3 import \
    nnUNetTrainerV2_DA3
from torch.optim import lr_scheduler


class nnUNetTrainerV2_Adam_Cosine_DA3(nnUNetTrainerV2_DA3):
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 5e-4
        self.weight_decay = 0.05

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_num_epochs,
                                                            eta_min=self.initial_lr * 0.01, verbose=True)

    def maybe_update_lr(self, epoch=None):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.CosineAnnealingLR, lr_scheduler._LRScheduler))

            self.lr_scheduler.step()
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def on_epoch_end(self):
        nnUNetTrainer.on_epoch_end(self)
        continue_training = self.epoch < self.max_num_epochs
        return continue_training
