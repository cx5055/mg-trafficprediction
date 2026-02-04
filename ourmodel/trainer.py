import torch
import pandas as pd
import os
import time
import copy
import pynvml
import matplotlib.pyplot as plt
from lib.logger import get_logger
from lib.usedmetrics import all_metrics_torch
import pynvml
from torch.utils.tensorboard import SummaryWriter

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.batches_seen = 0
        self.meminfo = 0
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.logger.info("Argument: %r", args)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def prepare_batch(self, data: dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

        ds = self.args.dataset
        if ds == "all_traffic":
            traffic = [data['traffic_x'],
                       data['service1_x'],
                       data['service2_x'],
                       data['service3_x']]
            label = [data['traffic_y'].squeeze(1),
                     data['service1_y'].squeeze(1),
                     data['service2_y'].squeeze(1),
                     data['service3_y'].squeeze(1)]
        elif ds == "totaltraffic":
            traffic = data['traffic_x']
            label = data['traffic_y'].squeeze(1)
        elif ds == "servicetraffic_video":
            traffic = data['service1_x']
            label = data['service1_y'].squeeze(1)
        elif ds == "servicetraffic_IoT":
            traffic = data['service2_x']
            label = data['service2_y'].squeeze(1)
        elif ds == "servicetraffic_data":
            traffic = data['service3_x']
            label = data['service3_y'].squeeze(1)
        else:
            raise ValueError(f"Unknown dataset type: {ds}")

        distance = data['distance_x']
        population = data['population_x']
        topology = data['topology_x']
        return traffic, label, distance, population, topology

    def evaluate(self, epoch, loader: torch.utils.data.DataLoader, mode: str) -> float:
        assert mode in ("Val", "Test")
        self.model.eval()
        total_loss = 0.0
        t0 = time.time()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                traffic, label, dist, pop, topo = self.prepare_batch(data)
                output = self.model(traffic, dist, pop)

                loss = self.loss(output, label)
                total_loss += loss.item()
            epoch_avg_loss = total_loss / len(loader)
            self.logger.info(
                f'********{mode} Epoch {epoch}: average Loss: {epoch_avg_loss:.6f}, time: {time.time() - t0:.2f} s'
            )
            return epoch_avg_loss

    def val_epoch(self, epoch: int) -> float:
        loader = self.val_loader if self.val_loader is not None else self.test_loader
        return self.evaluate(epoch, loader, "Val")

    def test_epoch(self, epoch: int) -> float:
        return self.evaluate(epoch, self.test_loader, "Test")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            self.batches_seen += 1
            traffic, label, dist, pop, topo = self.prepare_batch(data)
            self.optimizer.zero_grad()
            output = self.model(traffic, dist, pop)
            loss = self.loss(output, label)
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        train_epoch_loss = total_loss / self.train_per_epoch
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Train/LR', current_lr, epoch)
        self.writer.add_scalar('Train/Loss', train_epoch_loss, epoch)
        used_gb = (meminfo.used - self.meminfo.used) / 1024 ** 3
        self.writer.add_scalar('Train/GPU_Used_GB', used_gb, epoch)
        self.logger.info(f"Epoch {epoch} Learning Rate: {current_lr:.6e}")
        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, GPU cost: {:.2f} GB, train time: {:.2f} s'.format(
                epoch, train_epoch_loss, used_gb, time.time() - epoch_time
            )
        )

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def plot_loss_curve(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train')
        plt.plot(epochs, self.val_losses, label='Val')
        plt.plot(epochs, self.test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_figure_path)
        self.logger.info(f'Saved loss curve to {self.loss_figure_path}')

    def train(self):
        self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.logger.info("Model architecture:\n%s", self.model)
        best_model_state = None
        best_test_model_state = None
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        not_improved_count = 0
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            val_loss = self.val_epoch(epoch)
            self.val_losses.append(val_loss)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            test_loss = self.test_epoch(epoch)
            self.test_losses.append(test_loss)
            self.writer.add_scalar('Test/Loss', test_loss, 0)
            if train_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Stopping training.')
                break
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                not_improved_count = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.logger.info(f'New best validation loss: {val_loss:.6f}. Saving model.')
            else:
                not_improved_count += 1
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_model_state = copy.deepcopy(self.model.state_dict())

        if not self.args.debug and best_model_state is not None:
            torch.save(best_model_state, self.best_path)
            self.logger.info(f"Saved best validation model to {self.best_path}")
        if not self.args.debug and best_test_model_state is not None:
            torch.save(best_test_model_state, self.best_test_path)
            self.logger.info(f"Saved best test model to {self.best_test_path}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info("Testing best validation model:")
            csv_path = os.path.join(
                self.args.log_dir,
                f"best_model_metrics_{self.args.model}.csv"
            )
            self.test(self.test_loader, csv_path)

        if best_test_model_state is not None:
            self.model.load_state_dict(best_test_model_state)
            self.logger.info("Testing best test model:")
            csv_path = os.path.join(
                self.args.log_dir,
                f"best_model_metrics_{self.args.model}.csv"
            )
            self.test(self.test_loader, csv_path)
        self.plot_loss_curve()
        self.writer.close()

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def test(self, data_loader: torch.utils.data.DataLoader, csv_path):
        save_dir = os.path.dirname(csv_path)
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        multi_pred = None
        multi_true = None

        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader, 1):
                traffic, label, dist, pop, topo = self.prepare_batch(data)
                output = self.model(traffic, dist, pop)

                if isinstance(output, list):
                    if multi_pred is None:
                        T = len(output)
                        multi_pred = [[] for _ in range(T)]
                        multi_true = [[] for _ in range(T)]
                    for k in range(T):
                        multi_pred[k].append(output[k])
                        multi_true[k].append(label[k])
                else:
                    if multi_pred is None:
                        multi_pred = []
                        multi_true = []
                    multi_pred.append(output)
                    multi_true.append(label)
        self.logger.info("Saved all input features to .pt files")
        records = []
        if isinstance(multi_pred[0], list):
            for k, (pred_list, true_list) in enumerate(zip(multi_pred, multi_true), start=1):
                y_pred_k = torch.cat(pred_list, dim=0)
                y_true_k = torch.cat(true_list, dim=0)
                torch.save(y_pred_k, os.path.join(save_dir, f"task{k}_pred.pt"))
                torch.save(y_true_k, os.path.join(save_dir, f"task{k}_true.pt"))
                self.logger.info(f"--- Task {k} Overall Metrics ---")
                raw_metrics = all_metrics_torch(y_pred_k, y_true_k)
                record = {}
                for name, value in raw_metrics.items():
                    record[name] = value.item() if hasattr(value, "item") else value
                record["task"] = k
                records.append(record)
                for name, v in record.items():
                    if name == "task":
                        continue
                    if name in ("MAPE", "SMAPE"):
                        self.logger.info(f"{name:>6s}: {v * 100:8.3f}%")
                    else:
                        self.logger.info(f"{name:>6s}: {v:8.6f}")
        else:
            y_pred = torch.cat(multi_pred, dim=0)
            y_true = torch.cat(multi_true, dim=0)
            torch.save(y_pred, os.path.join(save_dir, "pred.pt"))
            torch.save(y_true, os.path.join(save_dir, "true.pt"))
            self.logger.info("--- Single Task Overall Metrics ---")
            raw_metrics = all_metrics_torch(y_pred, y_true)
            record = {name: (value.item() if hasattr(value, "item") else value)
                      for name, value in raw_metrics.items()}
            record["task"] = 1
            records.append(record)

            for name, v in record.items():
                if name == "task":
                    continue
                if name in ("MAPE", "SMAPE"):
                    self.logger.info(f"{name:>6s}: {v * 100:8.3f}%")
                else:
                    self.logger.info(f"{name:>6s}: {v:8.6f}")
        df = pd.DataFrame(records)
        cols = ["task"] + [c for c in df.columns if c != "task"]
        df = df[cols]
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved all metrics to {csv_path}")

    def test_only(self, model, args, data_loader, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        multi_pred = None
        multi_true = None
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader, 1):
                traffic, label, dist, pop, topo = self.prepare_batch(data)
                output = model(traffic, dist, pop)
                if isinstance(output, list):
                    if multi_pred is None:
                        T = len(output)
                        multi_pred = [[] for _ in range(T)]
                        multi_true = [[] for _ in range(T)]
                    for k in range(T):
                        multi_pred[k].append(output[k])
                        multi_true[k].append(label[k])
                else:
                    if multi_pred is None:
                        multi_pred = []
                        multi_true = []
                    multi_pred.append(output)
                    multi_true.append(label)

        logger.info("Saved all input features to .pt files")
        records = []
        save_dir = args.log_dir
        if isinstance(multi_pred[0], list):
            y_pred_list = []
            y_true_list = []
            for k, (pred_list, true_list) in enumerate(zip(multi_pred, multi_true), start=1):
                y_pred_k = torch.cat(pred_list, dim=0)
                y_true_k = torch.cat(true_list, dim=0)
                y_pred_list.append(y_pred_k)
                y_true_list.append(y_true_k)
                torch.save(y_pred_k, os.path.join(save_dir, f"task{k}_pred.pt"))
                torch.save(y_true_k, os.path.join(save_dir, f"task{k}_true.pt"))
                logger.info(f"--- Task {k} Overall Metrics ---")
                raw_metrics = all_metrics_torch(y_pred_k, y_true_k)
                record = {}
                for name, value in raw_metrics.items():
                    record[name] = value.item() if hasattr(value, "item") else value
                record["task"] = k
                records.append(record)
                for name, v in record.items():
                    if name == "task":
                        continue
                    if name in ("MAPE", "SMAPE"):
                        logger.info(f"{name:>6s}: {v * 100:8.3f}%")
                    else:
                        logger.info(f"{name:>6s}: {v:8.6f}")
            return y_pred_list, y_true_list, records
        else:
            y_pred = torch.cat(multi_pred, dim=0)
            y_true = torch.cat(multi_true, dim=0)
            torch.save(y_pred, os.path.join(args.log_dir, "pred.pt"))
            torch.save(y_true, os.path.join(args.log_dir, "true.pt"))
            logger.info("--- Single Task Overall Metrics ---")
            raw_metrics = all_metrics_torch(y_pred, y_true)
            record = {name: (value.item() if hasattr(value, "item") else value)
                      for name, value in raw_metrics.items()}
            record["task"] = 1
            records.append(record)

            for name, v in record.items():
                if name == "task":
                    continue
                if name in ("MAPE", "SMAPE"):
                    logger.info(f"{name:>6s}: {v * 100:8.3f}%")
                else:
                    logger.info(f"{name:>6s}: {v:8.6f}")
            return y_pred, y_true, records

        df = pd.DataFrame(records)
        cols = ["task"] + [c for c in df.columns if c != "task"]
        df = df[cols]
        csv_path = os.path.join(args.log_dir, f"Test_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved all metrics to {csv_path}")
