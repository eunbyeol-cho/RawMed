import os, tqdm, logging, wandb, csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from ehrsyn.criterions import ReconCriterion, ARCriterion
import ehrsyn.utils.trainer_utils as utils
from ehrsyn.datamodules import ehrsyn_data_loader, ehrtimelatent_data_loader
import ehrsyn.modules
import ehrsyn.criterions

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ehrsyn.modules.build_model(config)
        self.model = nn.DataParallel(model).to(self.device)
        total_parameters, trainable_parameters = utils.count_parameters(self.model)
        logger.info(self.model)
        logger.info(f"Total Parameters: {total_parameters:,} (Trainable: {trainable_parameters:,})")

        # Data Loaders
        if "AE" in self.config["exp_name"]:
            self.data_loaders = ehrsyn_data_loader(config)
        elif "AR" in self.config["exp_name"]:
            self.data_loaders = ehrtimelatent_data_loader(config)
        else:
            raise AssertionError("?")

        # Training options
        self.n_epochs = config['n_epochs']
        self.lr = config['lr']
        self.patience = config['patience']
        self.criterion = self._select_criterion(config)
        self.optimizer = self._select_optimizer(config, model)
        self.scheduler_type = self.config['scheduler_type']
        self.scheduler = self._select_scheduler(config)

        self.ckpt_name, self.path = utils.ExperimentCheckpoint(config).generate_checkpoint_path()
        logger.info(self.path)
        
        if not self.config["test_only"]:
            # self._initialize_wandb(config, self.ckpt_name)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            utils.save_config(config, self.path)
        
        self.c_epoch = 0
        if config["resume"]:
            self.c_epoch, self.model, self.optimizer = utils.load_model(self.path, self.model, self.optimizer)
            logger.info(f"Resume from {self.c_epoch}")

        if self.config["pretrained_AE_path"]:
            AE_config = utils.load_config(self.config["pretrained_AE_path"])
            event_autoencoder = ehrsyn.modules.build_model(AE_config)
            optimizer = self._select_optimizer(AE_config, event_autoencoder)
            _, event_autoencoder, _ = utils.load_model(self.config["pretrained_AE_path"], event_autoencoder, optimizer)
            self.event_autoencoder = nn.DataParallel(event_autoencoder).to(self.device)
            self.event_autoencoder.eval()

    def _select_criterion(self, config):
        if "AR" in config["exp_name"]:
            return ARCriterion(config)
        return ReconCriterion(config)

    def _select_optimizer(self, config, model):
        if config["optimizer"] == "Adam":
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        elif config["optimizer"] == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=self.lr)
        raise ValueError("Unsupported optimizer type")

    def _select_scheduler(self, config):
        if self.scheduler_type is not None:
            if self.scheduler_type == "steplr":
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config["scheduler_step_size"],
                    gamma=config["scheduler_gamma"]
                )
            else:
                raise ValueError("Unknown scheduler type")
        else:
            return None

    def _initialize_wandb(self, config, ckpt_name):
        wandb.init(
            project=config['wandb_project_name'],
            entity="emrsyn",
            config=config,
            reinit=True
        )
        wandb.run.name = ckpt_name
                    
    def train(self):
        
        self.early_stopping = utils.EarlyStopping(
            patience=self.patience, 
            compare=self.criterion.compare,
            metric=self.criterion.update_target
            )

        for epoch in range(self.c_epoch, self.n_epochs):
            self.model.train()
            
            for sample in tqdm.tqdm(self.data_loaders['train']):
                self.optimizer.zero_grad(set_to_none=True)
                net_output, targets = self.model(**sample['net_input'])
                
                loss = self.criterion('loss', net_output, targets)
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)

                self.optimizer.step()
                
                with torch.no_grad():
                    acc = self.criterion('acc', net_output, targets)

            with torch.no_grad():
                epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders['train']))

            summary = utils.log_from_dict(epoch_log, 'train', epoch)
            if not self.config['debug']:
                wandb.log(summary)

            if self.scheduler_type is not None:
                self.scheduler.step()
            
            should_stop = self.validate(epoch)
            if should_stop:
                break

        
        self.test()
        if not self.config['debug']:
            wandb.finish(0)

    def inference(self, epoch, subset, saver=None):
        self.model.eval()

        with torch.no_grad():
            for sample in tqdm.tqdm(self.data_loaders[subset]):
            
                net_output, targets = self.model(**sample['net_input'])
                if saver:
                    saver.concat(net_output, targets)
                loss = self.criterion('loss', net_output, targets)
                acc = self.criterion('acc', net_output, targets)

            epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders[subset]))
            summary = utils.log_from_dict(epoch_log, subset, epoch)
            if not self.config['debug']:
                wandb.log(summary)

        return epoch_log, saver

    def validate(self, epoch):
        break_token = False
        epoch_log, _ = self.inference(epoch, 'valid')
        
        if self.early_stopping(epoch_log[self.criterion.update_target]):
            utils.save_model(self.path, self.model, self.optimizer, epoch)

        if self.early_stopping.early_stop:
            logger.info(f'Early stopped! All valid finished at {epoch} ...')
            break_token=True
        return break_token

    def test(self, saver=None):
        epoch, self.model, _ = utils.load_model(self.path, self.model, self.optimizer)

        saver = utils.SaveNumpy(self.config, self.ckpt_name)
        test_subsets = self.config["test_subsets"].replace(" ", "").split(",")

        for subset in test_subsets: 
            logger.info(f'Test on {subset}!')
            epoch_log, saver = self.inference(epoch, subset, saver)      
                 
        save_dir = saver.save()
        return epoch_log
    
    def inference_for_comprehensive_test(self, epoch, subset, saver):
        self.model.eval()
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm.tqdm(self.data_loaders[subset])):
            
                net_output, targets = self.model(**sample['net_input'])
                
                if self.config["pretrained_AE_path"]:
                    self.event_autoencoder.eval()
                    net_output = self.event_autoencoder.module.decode(net_output)                 
                                        
                saver.concat(net_output, targets)

                if (idx+1) * self.config["batch_size"] >= self.config["gen_samples"]:
                    return saver

        return saver

    def comprehensive_test(self):
        epoch, self.model, _ = utils.load_model(self.path, self.model, self.optimizer)
        saver = utils.SaveNumpy(self.config, self.ckpt_name)
        test_subsets = ["total"]
        
        for subset in test_subsets: 
            logger.info(f'Test on {subset}!')
            if self.config["sample"]:
                logger.info(f'Sample on topk={self.config["topk"]} & temp={self.config["temperature"]}!')
                logger.info(f'{self.config["gen_samples"]} samples will be genereated')

            saver = self.inference_for_comprehensive_test(epoch, subset, saver)

        save_dir = saver.save()