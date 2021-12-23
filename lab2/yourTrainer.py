# https://github.com/Intelligent-Systems-Lab/FedML/blob/master/fedml_api/standalone/fedavg/my_model_trainer_classification.py

import logging
import torch
from torch import nn
from fedml_core.trainer.model_trainer import ModelTrainer


class ShaTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        # something magic here
        pass

    def test(self, test_data, device, args):
        # something magic here
        pass
    
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
