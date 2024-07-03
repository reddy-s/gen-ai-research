import logging
import torch
import os

os.environ["TERMINAL_MODE"] = "1"

from aicraft.utils.loaders import CartoonFacesLoader
from aicraft.models.vae.faces.model import FaceVAE
from aicraft.utils.loss import CustomLosses
from aicraft.models.AutoEncoderTrainers import VaeTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(42)


def train(
    model: FaceVAE,
    tl: torch.utils.data.DataLoader,
    vl: torch.utils.data.DataLoader,
    device: torch.device = None,
):
    hyperparameters = {
        "lr": 0.0005,
        "epochs": 10,
        "batch_size": tl.batch_size,
        "optimizer": "Adam",
        "scheduler": "ExponentialLR",
        "step_size": 10,
        "gamma": 0.97,
    }
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=hyperparameters["gamma"])
    VaeTrainer.train(
        tl=tl,
        vl=vl,
        model=model,
        loss_fn=CustomLosses.vae_loss_with_rmse,
        optimiser=optimiser,
        scheduler=exponential_scheduler,
        epochs=hyperparameters["epochs"],
        device=device,
    )

    logger.info("ℹ️ Saving model...")
    model.save(path="model-artefacts/face-vae-512-e40-tl.safetensors")
    logger.info("ℹ️ Successfully saved model")


if __name__ == "__main__":
    os.environ["TERMINAL_MODE"] = "1"

    if torch.cuda.is_available():
        logger.info("ℹ️ Using GPU")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("ℹ️ Using MPS")
        device = torch.device("mps")
    else:
        logger.info("ℹ️ Using CPU")
        device = torch.device("cpu")

    logger.info("ℹ️ Initialising model...")
    faceVAE = FaceVAE(embedding_size=512)
    faceVAE.load("model-artefacts/face-vae-512-e40.safetensors")

    logger.info("ℹ️ Initialising data loaders...")
    data_loader = CartoonFacesLoader(
        path="data/cartoon-faces", val_split=0.1, test_split=0.2
    )
    train_loader, val_loader, test_loader = data_loader.get_loaders(batch_size=256)

    logger.info("ℹ️ Training model...")
    train(faceVAE, test_loader, val_loader, device)
