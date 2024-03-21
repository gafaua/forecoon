from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import wandb
from lib.models.lstm_date_predictor import LSTMDate

from lib.trainers.base import BaseTrainer
import torch.nn as nn


class TimeSeriesDateTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        self.model = LSTMDate(
            69,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            output_size=2,#len(args.labels_output),
            query_size=67
        ).to(args.device)

        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="time-series-date",
                name=args.run_name,
                config=args.__dict__,
            )
        
        self.reg_criterion = nn.MSELoss()
        self.num_reg = 2
        self.output_size = 2#len(args.labels_output)


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval),
                      reg=deque(maxlen=self.log_interval))

        for i, batch in enumerate(pbar):
            self.opt.zero_grad()

            inp, outs = batch[0].to(self.device), batch[1].to(self.device)

            query, outs = outs[:, :-self.output_size], outs[:, -self.output_size:]
            preds = self.model(inp, query)
            loss_reg = self.reg_criterion(preds[:,:self.num_reg], outs[:,:self.num_reg])

            loss = loss_reg

            loss.backward()
            self.opt.step()

            self.step += 1
            losses["loss"].append(loss.item())
            losses["reg"].append(loss_reg.item())

            avg = {f"tr_{key}": np.mean(val) for key, val in losses.items()}
            pbar.set_postfix(dict(loss=loss.item(), **avg))

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data=avg, step=self.step)

        self.lr_scheduler.step()

    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list(),
                      reg=list())

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                inp, outs = batch[0].to(self.device), batch[1].to(self.device)

                query, outs = outs[:, :-self.output_size], outs[:, -self.output_size:]
                preds = self.model(inp, query)
                loss_reg = self.reg_criterion(preds[:,:self.num_reg], outs[:,:self.num_reg])

                loss = loss_reg

                losses["loss"].append(loss.item())
                losses["reg"].append(loss_reg.item())

                avg = {f"ev_{key}": np.mean(val) for key, val in losses.items()}
                pbar.set_postfix(dict(loss=loss.item(), **avg))

        if self.use_wandb:
            wandb.log(data=avg, step=self.step)
