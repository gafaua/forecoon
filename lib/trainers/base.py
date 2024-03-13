from abc import abstractmethod
from os import makedirs
import torch
from tqdm import tqdm
import wandb


class BaseTrainer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 args) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = args.device

        self.batch_size = args.batch_size
        self.lr = args.lr
        #self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.num_workers = args.num_workers

        self.use_wandb = args.use_wandb
        self.log_interval = args.log_interval

        self.save_dir = f"{args.save_dir}/{args.experiment}/{args.run_name}"
        makedirs(self.save_dir, exist_ok=True)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs)

        if args.checkpoint is not None:
            self._load_checkpoint(args.checkpoint)
        else:
            self.model_params = list(self.model.parameters())
            self.name = args.run_name
            self.step = 0
            self.epoch = 0

        print(f"Trainer ready, model with {sum(p.numel() for p in self.model_params):,} parameters")

    def train(self):
        train_epochs = range(self.epoch, self.num_epochs)
        for _ in train_epochs:
            self._run_train_epoch()
            self._save_checkpoint()
            self._run_val_epoch()
            self.epoch += 1
            print(f"LR: {self.lr_scheduler.get_last_lr()[0]:.5f}")


    @abstractmethod
    def _run_train_epoch(self):
        ...


    @abstractmethod
    def _run_val_epoch(self):
        ...


    def _save_checkpoint(self):
        model_dict = self.model.state_dict()

        data = dict(
            model_dict=model_dict,
            opt_dict=self.opt.state_dict(),
            epoch=self.epoch,
            step=self.step,
            name=self.name,
        )

        torch.save(data, f"{self.save_dir}/checkpoint_{self.step}.pth")
        print(f"Checkpoint saved in {self.save_dir}/checkpoint_{self.step}.pth")


    def _load_checkpoint(self, path):
        data = torch.load(path)
        self.model.load_state_dict(data["model_dict"])
        self.model_params = list(self.model.parameters())
        self.opt.load_state_dict(data["opt_dict"])
        self.epoch = data["epoch"]
        self.step = data["step"]
        self.name = f"{data['name']}_resumed"

        print("="*100)
        print(f"Resuming training from checkpoint {path}")
        print("="*100)