import pytorch_lightning as pl
from config import TrainingConfig
from data import get_mnist_dataloaders
from model import LitAutoEncoder


def main() -> None:
    cfg = TrainingConfig()
    train_loader, _ = get_mnist_dataloaders(cfg.batch_size)

    model = LitAutoEncoder(learning_rate=cfg.learning_rate)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
