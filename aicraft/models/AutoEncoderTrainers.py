import torch
import os
import logging
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from typing import Callable, Union

if "TERMINAL_MODE" in os.environ:
    if os.environ["TERMINAL_MODE"] == "1":
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
else:
    from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)


class AutoEncoderTrainer:

    @staticmethod
    def train(
        tl: torch.utils.data.DataLoader,
        vl: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimiser: torch.optim.Optimizer,
        scheduler: Union[ExponentialLR, StepLR],
        epochs: int = 20,
        device: torch.device = None,
    ):
        if device is not None:
            model.to(device)

        for epoch in range(epochs):
            model.train()
            _total_train_loss = 0
            tk0 = tqdm(tl, total=int(len(tl)), desc="Auto-Encoder Training")
            for batch_idx, (data, _) in enumerate(tk0):
                if device is not None:
                    data = data.to(device)
                optimiser.zero_grad()
                recon = model(data)
                _loss = loss_fn(recon, data)
                _loss.backward()
                optimiser.step()
                _total_train_loss += _loss.item()
            scheduler.step()
            _train_loss = _total_train_loss / len(tl)
            _val_loss = AutoEncoderTrainer.validate(vl, model, loss_fn, device)
            logger.info(
                f"Epoch {epoch + 1}, Loss: {_train_loss}, Validation Loss: {_val_loss}"
            )
            logger.info(f"LRs for epoch {epoch + 1}: {scheduler.get_last_lr()}")

    @staticmethod
    def validate(
        vl: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        device: torch.device,
    ):
        if device is not None:
            model.to(device)

        model.eval()

        _total_val_loss = 0
        for batch in vl:
            data = batch[0]
            if device is not None:
                data = data.to(device)
            recon = model(data)
            _loss = loss_fn(recon, data)
            _total_val_loss += _loss.item()
        return _total_val_loss / len(vl)


class VaeTrainer:

    @staticmethod
    def train(
        tl: torch.utils.data.DataLoader,
        vl: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimiser: torch.optim.Optimizer,
        scheduler: Union[ExponentialLR, StepLR],
        epochs: int = 20,
        device: torch.device = None,
    ):
        if device is not None:
            model.to(device)

        for epoch in range(epochs):
            model.train()
            _total_train_loss = 0
            tk0 = tqdm(tl, total=int(len(tl)), desc="VAE Training")
            for batch_idx, (data, _) in enumerate(tk0):
                if device is not None:
                    data = data.to(device)
                optimiser.zero_grad()
                _mu, _sigma, recon = model(data)
                _loss, _recon_loss, _kld_loss = loss_fn(recon, data, _mu, _sigma)
                _loss.backward()
                optimiser.step()
                _total_train_loss += _loss.item()
            scheduler.step()
            _train_loss = _total_train_loss / len(tl)
            _val_loss = VaeTrainer.validate(vl, model, loss_fn, device)
            logger.info(
                f"Epoch {epoch + 1}, Loss: {_train_loss}, Validation Loss: {_val_loss}"
            )
            logger.info(f"LRs for epoch {epoch + 1}: {scheduler.get_last_lr()}")

    @staticmethod
    def validate(
        vl: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        device: torch.device,
    ):
        if device is not None:
            model.to(device)

        model.eval()

        _total_val_loss = 0
        for batch in vl:
            data = batch[0]
            if device is not None:
                data = data.to(device)
            _mu, _sigma, recon = model(data)
            _loss, _recon_loss, _kld_loss = loss_fn(recon, data, _mu, _sigma)
            _total_val_loss += _loss.item()
        return _total_val_loss / len(vl)
