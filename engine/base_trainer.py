import os
import torch
from thop import profile, clever_format
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import engine.utils as utils
from data.testdataset import TestDataset
from data.traindataset import TrainDataset
from engine.cyclic_lr import CyclicCosAnnealingLR
from engine.logger import get_logger


class BaseTrainer:
    """Base Class for training and evaluation.

            If you want to modify some methods in this class (eg. fit, eval, plot, check_trainer),
    please inherit this class and overload them.

    Use:
            Use Trainer.fit() to start training.
            Initialize the class with the configuration file. Trainer will automatically load
    configuration, save checkpoints, draw tensorboard, set optimize etc.
            If your training is break by accidents, re-run the Trainer.fit() to automatically restore
    snapshots (including models, optimizer, scheduler and training clock).
    """

    def __init__(self, config_name, default_config_name='default'):
        """Initialize class with specified configuration.
        Experiment configurations are loaded from config_name.
        Parameters that do not appear in config_name will be replaced with default parameters.
        """
        self.configs = utils.load_configs(config_name, default_config_name)
        self.experiment_path = f"{self.configs['experiment_config']['save_path']}/{self.configs['experiment_config']['exp_name']}"
        self.logger = get_logger(self.configs['experiment_config']['exp_name'], )
        self.snapshot = self.get_snapshot()
        self.model = self.get_model()
        self.optimizer, self.scheduler = self.get_optim()
        self.train_dataset = TrainDataset(self.configs['train_dataset_config'])
        self.test_dataset = TestDataset(self.configs['test_dataset_config'])
        self.loss_func = utils.load_loss(self.configs['training_config']['loss'], self.configs['loss_config'])
        self.writer = self.get_tsb()
        if self.configs['experiment_config']['check_trainer']:
            self.check_trainer()

    def fit(self):
        """Main training loop."""
        self.step, start = 0, 0
        max_epoch = self.configs['training_config']['max_epoch']
        batch_size = self.configs['training_config']['batch_size']
        if self.snapshot is not None:
            start = self.snapshot['epoch'] + 1
            self.step = self.snapshot['step']
        for self.epoch_i in range(start, max_epoch):
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            self.training_metrics = utils.ImageMetrics()
            self.loss_statistics = utils.LossStatistics()
            for self.batch_i, x in tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc=f'Training epoch:{self.epoch_i}',
                    leave=False):
                x['input'] = x['input'].to(self.device)
                x['gt'] = x['gt'].to(self.device)
                self.model.zero_grad()
                y = self.model(x)
                tensors = {**y, **x}
                loss = self.loss_func(tensors)
                loss['tot'].backward()
                self.loss_statistics.append(loss)
                self.optimizer.step()
                self.training_metrics.append(tensors)
                self.writer.add_scalar('train/epoch', self.epoch_i, self.step)
                self.step += 1
            self.plot()
            if self.scheduler is not None:
                self.scheduler.step(self.epoch_i)
            if self.epoch_i % self.configs['experiment_config']['saving_interval'] == 0:
                self.save('latest.pkl')
                self.save(f'{self.epoch_i}.pkl')
            if self.epoch_i % self.configs['experiment_config']['eval_interval'] == 0:
                self.eval(self.get_module().state_dict())

    def eval(self, state_dict):
        """Validating during train."""
        eval_model = utils.load_model(self.configs['training_config']['model'], self.configs['network_config'])
        eval_model = eval_model.to(self.device)
        eval_model.load_state_dict(state_dict)
        eval_model = eval_model.eval()
        with torch.no_grad():
            self.test_metrics = utils.ImageMetrics()
            self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)
            for self.batch_i, x in tqdm(
                    enumerate(self.test_dataloader),
                    total=len(self.test_dataloader),
                    desc=f'Evaluating epoch:{self.epoch_i}',
                    leave=False):
                # input_image = utils.split_test_image(input_image)
                x['input'] = x['input'].to(self.device)
                x['gt'] = x['gt'].to(self.device)
                y = eval_model(x)
                tensors = {**y, **x}
                self.test_metrics.append(tensors)
            test_metrics = self.test_metrics.eval()
            for i in test_metrics:
                self.writer.add_scalar(f'test/{i}', test_metrics[i], self.epoch_i)
            self.logger.info(f'Evaluating epoch:{self.epoch_i} {test_metrics}')

    def plot(self):
        """Plot parameters on tensorboard."""

        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        self.writer.add_scalar('train/lr', lr, self.epoch_i)
        training_metrics = self.training_metrics.eval()
        for i in training_metrics:
            self.writer.add_scalar(f'training/{i}', training_metrics[i], self.epoch_i)
        loss = self.loss_statistics.eval()
        for i in loss:
            self.writer.add_scalar(f'loss/{i}', loss[i], self.epoch_i)
        self.logger.info(f'Training epoch:{self.epoch_i} loss={loss}')
        for i, (name, net_param) in enumerate(self.model.named_parameters()):
            if name[:7] == 'module.':
                name = name[7:]
            name = name.replace('.', '/', 1)
            if 'relu' in name:
                self.writer.add_histogram(f'relu_{name}', net_param, self.epoch_i)
            elif 'bn' in name:
                self.writer.add_histogram(f'bn_{name}', net_param, self.epoch_i)
            else:
                self.writer.add_histogram(name, net_param, self.epoch_i)

    def save(self, name):
        """Save snapshot and checkpoint."""
        if self.scheduler is not None:
            torch.save({
                'epoch': self.epoch_i,
                'step': self.step,
                'model_state_dict': self.get_module().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, f"{self.experiment_path}/{name}")
        else:
            torch.save({
                'epoch': self.epoch_i,
                'step': self.step,
                'model_state_dict': self.get_module().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, f"{self.experiment_path}/{name}")

    def get_tsb(self):
        """Initialize tensorboard writer."""
        tsb_path = f"{self.configs['experiment_config']['save_path']}/{self.configs['experiment_config']['exp_name']}/train.events"
        writer = SummaryWriter(tsb_path)
        return writer

    def check_trainer(self):
        """Check the trainer before training."""
        self.logger.info('Checking test dataset.')
        x = self.test_dataset[0]
        self.logger.info('Checking train dataset.')
        dataloader = DataLoader(dataset=self.train_dataset, batch_size=1)
        for x in dataloader:
            x['input'] = x['input'].to(self.device)
            x['gt'] = x['gt'].to(self.device)
            self.logger.info('Checking model.')
            flops, params = profile(self.get_module(), inputs=(x,))
            cflops, cparams = clever_format([flops, params], "%.3f")
            self.logger.info(f'flops:{flops}({cflops}) params:{params}({cparams})')
            y = self.model(x)
            tensors = {**y, **x}
            self.logger.info('Checking loss.')
            loss = self.loss_func(tensors)
            self.logger.info(f'Check passed. Output shape is {y["output"].shape}. Loss is {loss}.')
            break

    def get_snapshot(self):
        """Get snapshot if exists."""
        snapshot_path = f"{self.experiment_path}/latest.pkl"
        if os.path.exists(snapshot_path):
            self.logger.warning(f"Snapshot is available, loading from {snapshot_path}.")
            snapshot = torch.load(snapshot_path)
        else:
            snapshot = None
        return snapshot

    def get_module(self):
        """Get nn.Module instead of nn.DataParallel."""
        if isinstance(self.model, nn.Module):
            return self.model
        elif isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            self.logger.critical('self.model must be a nn.Module or nn.DataParallel.')
            raise NotImplementedError

    def get_model(self):
        """Get model from configuration, load state dict if available and set gpu environment."""
        model = utils.load_model(self.configs['training_config']['model'], self.configs['network_config'])
        if self.snapshot is not None:
            model.load_state_dict(self.snapshot['model_state_dict'])
        if self.configs['gpu_config']['use_gpu']:
            self.device = torch.device("cuda")
            if utils.set_gpu(self.configs['gpu_config']) > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
        return model

    def get_optim(self):
        """Get optimizer and scheduler, and load state dict if available."""
        lr = self.configs['training_config']['lr']
        if type(self.model) is nn.DataParallel:
            optimizer = optim.Adam(self.get_module().parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if self.configs['training_config']['cyclic_lr']:
            self.logger.info('Enable cyclic learning rate.')
            scheduler = CyclicCosAnnealingLR(
                optimizer, milestones=[50, 100, 150, 200], decay_milestones=[50, 100, 150, 200], eta_min=1e-6)
        else:
            scheduler = None
        # scheduler = CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=self.configs['training_config']['max_epoch'])
        if self.snapshot is not None:
            optimizer.load_state_dict(self.snapshot['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(self.snapshot['scheduler_state_dict'])
        return optimizer, scheduler

    def change_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
