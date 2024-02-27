from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import wandb
from lib.models.simple_predictor import SimplePredictor
from lib.trainers.base import BaseTrainer
import torch.nn as nn


class ImageTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        self.model = SimplePredictor(args.backbone,
                                     args.dim,
                                     args.hidden_dim,
                                     3).to(args.device) # Grade classification [3,4,5]
        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="image",
                name=args.run_name,
                config=args.__dict__,
            )
        
        self.criterion = nn.CrossEntropyLoss()


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval))
        for batch in pbar:
            self.opt.zero_grad()

            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, labels)
            #print(torch.argmax(preds[0]).item(), labels[0].item())
            loss.backward()
            self.opt.step()
            self.step += 1
            losses["loss"].append(loss.item())
            running_avg = np.mean(losses["loss"])
            pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data={"train_loss": running_avg}, step=self.step)

        self.lr_scheduler.step(self.epoch)

    @torch.no_grad()
    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())
        for batch in pbar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, labels)

            losses["loss"].append(loss.item())
            running_avg = np.mean(losses["loss"])
            pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

        if self.use_wandb:
            wandb.log(data={"val_loss": np.mean(losses["loss"])}, step=self.step)
