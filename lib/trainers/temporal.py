from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import wandb
from lib.models.simple_predictor import SimplePredictor
from lib.models.temporal_predictor import TemporalPredictor
from lib.trainers.base import BaseTrainer
import torch.nn as nn
from lib.utils.dataset import TemporalSequencePairDataset
from torch.utils.data import DataLoader

from lib.utils.evaluation import get_balance_accuracy_score, print_confusion_matrix


class TemporalTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        self.model = TemporalPredictor(args.backbone,
                                     args.dim,
                                     args.hidden_dim).to(args.device)

        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="temporal",
                name=args.run_name,
                config=args.__dict__,
            )

        self.criterion = nn.MSELoss()


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval))

        for i, sequence in enumerate(pbar):

            sequence_loader = DataLoader(sequence[0], self.batch_size, shuffle=True, num_workers=0)
            for batch in sequence_loader:
                self.opt.zero_grad()

                img1, img2, delta_time = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                #print(torch.mean(images[0]).item())
                preds = self.model(img1, img2).squeeze()
                if img1.shape[0] == 1:
                    preds = preds.unsqueeze(0)

                loss = self.criterion(preds, delta_time.type(torch.float32))

                loss.backward()
                self.opt.step()

                self.step += 1
                losses["loss"].append(loss.item())
                running_avg = np.mean(losses["loss"])
                pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

                if self.use_wandb and self.step % self.log_interval == 0:
                    wandb.log(data={"train_loss": running_avg}, step=self.step)

        self.lr_scheduler.step()


    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())

        with torch.no_grad():
            for i, sequence in enumerate(pbar):

                sequence_loader = DataLoader(sequence[0], self.batch_size, shuffle=True, num_workers=0)
                for batch in sequence_loader:
                    img1, img2, delta_time = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                    #print(torch.mean(images[0]).item())
                    preds = self.model(img1, img2).squeeze()
                    if img1.shape[0] == 1:
                        preds = preds.unsqueeze(0)

                    loss = self.criterion(preds, delta_time.type(torch.float32))

                    losses["loss"].append(loss.item())
                    running_avg = np.mean(losses["loss"])
                    pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

        if self.use_wandb:
            wandb.log(data={"val_loss": np.mean(losses["loss"])}, step=self.step)

