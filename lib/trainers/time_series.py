from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import wandb
from lib.models.lstm_predictor import LSTM

from lib.trainers.base import BaseTrainer
import torch.nn as nn


class TimeSeriesTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        self.model = LSTM(
            69,
            hidden_size=args.hidden_dim,
            num_layers=2,
            output_size=2
        ).to(args.device)

        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="time-series",
                name=args.run_name,
                config=args.__dict__,
            )
        
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.num_reg = 2


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval),
                      #cls=deque(maxlen=self.log_interval),
                      reg=deque(maxlen=self.log_interval))

        for i, batch in enumerate(pbar):
            self.opt.zero_grad()

            inp, outs = batch[0].to(self.device), batch[1].to(self.device)

            preds = self.model(inp).unsqueeze(1)

            loss_reg = self.reg_criterion(preds[:,:,:self.num_reg], outs[:,:,:self.num_reg])
            #loss_cls = self.cls_criterion(preds[:,:,self.num_reg:].squeeze(), torch.argmax(outs[:,:,self.num_reg:], dim=2).squeeze())

            loss = loss_reg #+ loss_cls

            loss.backward()
            self.opt.step()

            self.step += 1
            losses["loss"].append(loss.item())
            losses["reg"].append(loss_reg.item())
            #losses["cls"].append(loss_cls.item())
            avg = {f"tr_{key}": np.mean(val) for key, val in losses.items()}
            pbar.set_postfix(dict(loss=loss.item(), **avg))

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data=avg, step=self.step)

        self.lr_scheduler.step()

       # accuracy = get_balance_accuracy_score(all_preds, all_labels)
        # if self.use_wandb:
        #     wandb.log(data=dict(train_acc=accuracy))
        #print(f"Train Balanced Accuracy: {accuracy:.3f}")
        #print_confusion_matrix(all_preds, all_labels)

    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list(),
                      #cls=list(),
                      reg=list())

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                inp, outs = batch[0].to(self.device), batch[1].to(self.device)
                #print(torch.mean(images[0]).item())
                preds = self.model(inp).unsqueeze(1)
                loss_reg = self.reg_criterion(preds[:,:,:self.num_reg], outs[:,:,:self.num_reg])
                #loss_cls = self.cls_criterion(preds[:,:,self.num_reg:].squeeze(), torch.argmax(outs[:,:,self.num_reg:], dim=2).squeeze())

                loss = loss_reg #+ loss_cls

                losses["loss"].append(loss.item())
                losses["reg"].append(loss_reg.item())
                #losses["cls"].append(loss_cls.item())
                avg = {f"ev_{key}": np.mean(val) for key, val in losses.items()}
                pbar.set_postfix(dict(loss=loss.item(), **avg))

        if self.use_wandb:
            wandb.log(data=avg, step=self.step)
