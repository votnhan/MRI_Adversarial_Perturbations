from base import BaseTrainer
from utils import MetricTracker, save_output, save_mask2image
from utils.lr_scheduler import MyReduceLROnPlateau
import torch.nn as nn
import torch
import os


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, criterion, metrics, optimizer, config, lr_scheduler=None):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.lr_scheduler = lr_scheduler
        self.loss_name = 'supervised_loss'

        # Metrics
        # Train
        self.train_loss = MetricTracker(self.loss_name, self.writer)
        self.train_metrics = MetricTracker(*self.metric_names,
                                           self.writer)
        # Validation
        self.valid_loss = MetricTracker(self.loss_name, self.writer)
        self.valid_metrics = MetricTracker(*self.metric_names,
                                           self.writer)
        # Test
        self.test_loss = MetricTracker(self.loss_name, self.writer)
        self.test_metrics = MetricTracker(*self.metric_names,
                                          self.writer)

        if isinstance(self.model, nn.DataParallel):
            self.criterion = nn.DataParallel(self.criterion)

        # Resume checkpoint if path is available in config
        cp_path = self.config['trainer'].get('resume_path')
        if cp_path:
            super()._resume_checkpoint()

    def reset_scheduler(self):
        self.train_loss.reset()
        self.train_metrics.reset()
        self.valid_loss.reset()
        self.valid_metrics.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
            self.lr_scheduler.reset()

    def prepare_train_epoch(self, epoch):
        self.logger.info('EPOCH: {}'.format(epoch))
        self.reset_scheduler()

    def _train_epoch(self, epoch):
        self.model.train()
        self.prepare_train_epoch(epoch)
        for batch_idx, (data, target, image_name) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            # For debug model
            if torch.isnan(loss):
                super()._save_checkpoint(epoch)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update train loss, metrics
            self.train_loss.update(self.loss_name, loss.item())
            for metric in self.metrics:
                self.train_metrics.update(metric.__name__, metric(output, target), n=output.shape[0])

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

        # step lr scheduler
        if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
            self.lr_scheduler.step(self.valid_loss.avg(self.loss_name))

        return log

    @staticmethod
    def get_metric_message(metrics, metric_names):
        metrics_avg = [metrics.avg(name) for name in metric_names]
        message_metrics = ', '.join(['{}: {:.6f}'.format(x, y) for x, y in zip(metric_names, metrics_avg)])
        return message_metrics

    def log_for_step(self, epoch, batch_idx):
        message_loss = 'Train Epoch: {} [{}]/[{}] Dice Loss: {:.6f}'.format(epoch, batch_idx, self.len_epoch,
                                                                            self.train_loss.avg(self.loss_name))

        message_metrics = SegmentationTrainer.get_metric_message(self.train_metrics, self.metric_names)
        self.logger.info(message_loss)
        self.logger.info(message_metrics)

    def _valid_epoch(self, epoch, save_result=False, save_for_visual=False):
        self.model.eval()
        self.valid_loss.reset()
        self.valid_metrics.reset()
        self.logger.info('Validation: ')
        with torch.no_grad():
            for batch_idx, (data, target, image_name) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_loss.update(self.loss_name, loss.item())
                for metric in self.metrics:
                    self.valid_metrics.update(metric.__name__, metric(output, target), n=output.shape[0])

                if save_result:
                    save_output(output, image_name, epoch, self.checkpoint_dir, percent=1)

                if save_for_visual:
                    save_mask2image(output, image_name, os.path.join(self.checkpoint_dir, 'output'))
                    save_mask2image(target, image_name, os.path.join(self.checkpoint_dir, 'target'))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('{}/{}'.format(batch_idx, len(self.valid_data_loader)))
                    self.logger.debug('{}: {}'.format(self.loss_name, self.valid_loss.avg(self.loss_name)))
                    self.logger.debug(SegmentationTrainer.get_metric_message(self.valid_metrics, self.metric_names))

        log = self.valid_loss.result()
        log.update(self.valid_metrics.result())
        val_log = {'val_{}'.format(k): v for k, v in log.items()}
        return val_log

    def _test_epoch(self, epoch, save_result=False, save_for_visual=False):
        self.model.eval()
        self.test_loss.reset()
        self.test_metrics.reset()
        self.logger.info('Test: ')
        with torch.no_grad():
            for batch_idx, (data, target, image_name) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                self.test_loss.update(self.loss_name, loss.item())
                for metric in self.metrics:
                    self.test_metrics.update(metric.__name__, metric(output, target), n=output.shape[0])

                if save_result:
                    save_output(output, image_name, epoch, self.checkpoint_dir, percent=1)

                if save_for_visual:
                    save_mask2image(output, image_name, os.path.join(self.checkpoint_dir, 'output'))
                    save_mask2image(target, image_name, os.path.join(self.checkpoint_dir, 'target'))

                self.logger.debug('{}/{}'.format(batch_idx, len(self.test_data_loader)))
                self.logger.debug('{}: {}'.format(self.loss_name, self.test_loss.avg(self.loss_name)))
                self.logger.debug(SegmentationTrainer.get_metric_message(self.test_metrics, self.metric_names))

        log = self.test_loss.result()
        log.update(self.test_metrics.result())
        test_log = {'test_{}'.format(k): v for k, v in log.items()}
        return test_log

