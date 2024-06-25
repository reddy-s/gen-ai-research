import torch
import os
from typing import Callable, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import logging
from uuid import uuid4
from torch.optim.lr_scheduler import ExponentialLR
from .utils.average_meter import AverageMeter

if "TERMINAL_MODE" in os.environ:
    if os.environ["TERMINAL_MODE"] == "1":
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
else:
    from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)


class TorchRunner:
    @staticmethod
    def training_step(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
    ):
        """
        A method to perform a single training step using the provided data loader, model, loss function, optimizer, and device.

        Parameters:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (Callable): The loss function to calculate the loss.
            optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
            device (torch.device, optional): The device to run the training on. Defaults to None.

        Returns:
            float: The average loss incurred during the training step.
        """
        # meter
        loss = AverageMeter()
        # switch to train mode
        if device is not None:
            model = model.to(device)
        model.train()

        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        for batch_idx, (data, target) in enumerate(tk0):
            if device is not None:
                data, target = data.to(device), target.to(device)

            # initialize the optimizer
            optimizer.zero_grad()

            if model.__class__.__name__ == "AutoEncoder":
                recon = model.forward(data)
                loss_this = loss_fn(recon, data)
            elif model.__class__.__name__ == "VariationalAutoEncoder":
                mu, sigma, recon = model.forward(data)
                loss_this, _, _ = loss_fn(recon, data, mu, sigma)
            else:
                raise ValueError(
                    "Unknown model type. Can only train AutoEncoder or VariationalAutoEncoder."
                )

            # compute the backward pass
            loss_this.backward()
            # update the parameters
            optimizer.step()
            # update the loss meter
            loss.update(loss_this.item(), target.shape[0])
        logger.info("Train: Average loss: {:.4f}".format(loss.avg))
        return loss.avg

    @staticmethod
    def train(
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: ExponentialLR,
        epochs: int = 10,
        device: torch.device = None,
        runs_folder: str = "../checkpoints/runs",
    ):
        """
        Train the model using the given data loaders, model, loss function, optimizer, and scheduler.

        Parameters:
            train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (Callable): The loss function to optimize.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            scheduler (Union[ExponentialLR, ReduceLROnPlateau], optional): The learning rate scheduler. Defaults to None.
            epochs (int, optional): The number of epochs to train. Defaults to 10.
            device (torch.device, optional): The device to run training on. Defaults to None.
            runs_folder (str, optional): The folder path to save training checkpoints. Defaults to "../checkpoints/runs".
        """
        writer = SummaryWriter(
            f"{runs_folder}/{model.__class__.__name__}/{uuid4().hex}"
        )
        writer.add_graph(model.to(device), next(iter(train_loader))[0].to(device))
        validation_loss = None

        # TODO: register hyper-parameters
        for epoch in range(1, epochs + 1):
            logger.info(
                "Epoch: {}/{}\n=====================================".format(
                    epoch, epochs
                )
            )

            training_loss = TorchRunner.training_step(
                train_loader, model, loss_fn, optimizer, device
            )
            validation_loss = TorchRunner.test(val_loader, model, loss_fn, device)

            scheduler.step()

            writer.add_scalar(
                tag="training_loss", scalar_value=training_loss, global_step=epoch
            )
            writer.add_scalar(
                tag="validation_loss",
                scalar_value=validation_loss,
                global_step=epoch,
            )
            logger.info(
                "LRs for epoch {}: {}\n".format(epoch, str(scheduler.get_last_lr()))
            )

        writer.add_hparams(
            hparam_dict={
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.defaults["lr"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "optimizer": optimizer.__class__.__name__,
                "loss_fn": loss_fn.__name__,
                "schedular": scheduler.__class__.__name__,
                "epochs": epochs,
                "model": model.__class__.__name__,
                "device": device.type,
            },
            metric_dict={"validation_loss": validation_loss},
        )

    @staticmethod
    def test(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        device: torch.device = None,
    ) -> float:
        """
        Calculate the loss and accuracy metrics for the given DataLoader using the provided model and loss function.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader containing the data.
            model (torch.nn.Module): The model used for prediction.
            loss_fn (Callable): The loss function used for calculating the loss.
            device (torch.device, optional): The device to run the model on. Defaults to None.
            self_supervised: (bool, optional): Whether the model is self-supervised. Defaults to False.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and accuracy.
        """
        loss = AverageMeter()
        if device is not None:
            model = model.to(device)
        # switch to test mode
        model.eval()
        for data, target in loader:
            if device is not None:
                data, target = data.to(device), target.to(
                    device
                )  # data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                # compute the forward pass
                # it can also be achieved by model.forward(data)
                if model.__class__.__name__ == "AutoEncoder":
                    output = model.forward(data)
                elif model.__class__.__name__ == "VariationalAutoEncoder":
                    mu, sigma, output = model.forward(data)
                else:
                    raise ValueError(
                        "Unknown model type. Can only train AutoEncoder or VariationalAutoEncoder."
                    )

            if model.__class__.__name__ == "AutoEncoder":
                loss_this = loss_fn(output, data)
                loss.update(loss_this.item(), target.shape[0])
            elif model.__class__.__name__ == "VariationalAutoEncoder":
                loss_this = loss_fn(output, data, mu, sigma)
                loss.update(loss_this.item(), target.shape[0])
            else:
                raise ValueError(
                    "Unknown model type. Can only train AutoEncoder or VariationalAutoEncoder."
                )

        logger.info("Validation/Test: Average loss: {:.4f}".format(loss.avg))

        return loss.avg

    @staticmethod
    def get_summary(model: torch.nn.Module, shape: Tuple[int, int, int]) -> None:
        """
        A static method to get a summary of a PyTorch model given its architecture shape.

        Parameters:
            model (torch.nn.Module): The PyTorch model to summarize.
            shape (Tuple[int, int, int]): The shape of the model's architecture.

        Returns:
            None
        """
        summary(model, shape)
