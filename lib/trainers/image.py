from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import wandb
from lib.models.simple_predictor import SimplePredictor
from lib.trainers.base import BaseTrainer
import torch.nn as nn

from lib.utils.evaluation import get_balance_accuracy_score, print_confusion_matrix


class ImageTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        self.model = SimplePredictor(args.backbone,
                                     args.dim,
                                     args.hidden_dim,
                                     4).to(args.device) # Grade classification [3,4,5]
        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="image",
                name=args.run_name,
                config=args.__dict__,
            )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    def _run_train_epoch(self):
        # TODO context manager for metrics
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval))

        all_preds = []
        all_labels = []

        for i, batch in enumerate(pbar):
            self.opt.zero_grad()

            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            #print(torch.mean(images[0]).item())
            preds = self.model(images)
            loss = self.criterion(preds, labels)

            loss.backward()
            self.opt.step()

            self.step += 1
            losses["loss"].append(loss.item())
            running_avg = np.mean(losses["loss"])
            pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

            all_preds.append(torch.argmax(preds, dim=1).cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data={"train_loss": running_avg}, step=self.step)

        self.lr_scheduler.step()

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = get_balance_accuracy_score(all_preds, all_labels)
        if self.use_wandb:
            wandb.log(data=dict(train_acc=accuracy))
        print(f"Train Balanced Accuracy: {accuracy:.3f}")
        print_confusion_matrix(all_preds, all_labels)

    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, labels)

                all_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                losses["loss"].append(loss.item())
                running_avg = np.mean(losses["loss"])
                pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = get_balance_accuracy_score(all_preds, all_labels)

        if self.use_wandb:
            wandb.log(data={"val_loss": np.mean(losses["loss"]), "val_acc": accuracy}, step=self.step)

        print(f"Eval Balanced Accuracy: {accuracy:.3f}")
        print_confusion_matrix(all_preds, all_labels)
