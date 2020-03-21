from .segmentation_trainer import SegmentationTrainer
from utils import save_output, inf_norm_adjust
import torch
import sys


class AdversarialTrainer(SegmentationTrainer):
    def __init__(self, generator, trained_model, criterion, metrics, optimizer, config, lr_scheduler=None):
        super().__init__(generator, criterion, metrics, optimizer, config, lr_scheduler)
        self.trained_model = trained_model
        self.load_trained_model()
        self.freeze_trained_model()
        self.noise_epsilon = self.config['trainer']['noise_eps']
        self.range_input = self.config['transforms']['image_transforms']['range_scale']

    def freeze_trained_model(self):
        for p in self.trained_model.parameters():
            p.require_grad = False

    def load_trained_model(self):
        snapshot_path = self.config['pre_trained']['snapshot']
        checkpoint = torch.load(snapshot_path)
        self.trained_model.load_state_dict(checkpoint['state_dict'])

    def _train_epoch(self, epoch):
        self.model.train()
        self.trained_model.eval()
        self.prepare_train_epoch(epoch)
        for batch_idx, (data, target, image_name) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            noise = self.model(data)
            # TODO: control noise
            noise_clamped = inf_norm_adjust(noise, self.noise_epsilon)
            noise_input = noise_clamped + data
            # TODO: control noise input
            noise_input_clamped = torch.clamp(noise_input, self.range_input[0], self.range_input[1])
            output = self.trained_model(noise_input_clamped)
            loss = (-1)*self.criterion(output, target)
            # For debug model
            if torch.isnan(loss):
                super()._save_checkpoint(epoch)
                sys.exit(0)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update train loss, metrics
            self.train_loss.update(self.loss_name, loss.item())
            for metric in self.metrics:
                self.train_metrics.update(metric.__name__, metric(output, target))

            if batch_idx % self.log_step == 0:
                self.log_for_step(epoch, batch_idx)

            if self.save_for_track and (batch_idx % self.save_for_track == 0):
                save_output(output, image_name, epoch, self.checkpoint_dir)

            if batch_idx == self.len_epoch:
                break

        log = self.train_loss.result()
        log.update(self.train_metrics.result())

        if self.do_validation and (epoch % self.do_validation_interval == 0):
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

    def _valid_epoch(self, epoch):
        self.trained_model.eval()
        self.model.eval()
        self.valid_loss.reset()
        self.valid_metrics.reset()
        self.logger.info('Validation: ')
        with torch.no_grad():
            for batch_idx, (data, target, image_name) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                noise = self.model(data)
                # TODO: control noise
                noise_clamped = inf_norm_adjust(noise, self.noise_epsilon)
                noise_input = noise_clamped + data
                # TODO: control noise input
                noise_input_clamped = torch.clamp(noise_input, self.range_input[0], self.range_input[1])
                output = self.trained_model(noise_input_clamped)
                loss = (-1) * self.criterion(output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_loss.update(self.loss_name, loss.item())
                for metric in self.metrics:
                    self.valid_metrics.update(metric.__name__, metric(output, target))

                self.logger.debug('{}/{}'.format(batch_idx, len(self.valid_data_loader)))
                self.logger.debug('{}: {}'.format(self.loss_name, self.valid_loss.avg(self.loss_name)))
                self.logger.debug(SegmentationTrainer.get_metric_message(self.valid_metrics, self.metric_names))

        log = self.valid_loss.result()
        log.update(self.valid_metrics.result())
        val_log = {'val_{}'.format(k): v for k, v in log.items()}
        return val_log
