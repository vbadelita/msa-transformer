from typing import Optional
import sys
from pathlib import Path
import logging
import pytorch_lightning as pl
import torch
import esm
from evo.dataset import (
    RandomCropDataset,
    MaskedTokenWrapperDataset,
    EncodedFastaDataset,
    BatchBySequenceLength,
)
from evo.tokenization import Vocab
from model import ESM1b, TransformerConfig, OptimizerConfig
from dataset import TRRosettaContactDataset
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning.utilities import CombinedLoader


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

current_directory = Path(__file__).parent.absolute()


@dataclass
class DataConfig:
    train_fasta_path: str = str(current_directory / "data/trrosetta.fasta")
    valid_fasta_path: Optional[str] = None
    trrosetta_path: str = str(current_directory / "data" / "trrosetta")
    trrosetta_train_split: str = "valid_train.txt"
    trrosetta_valid_split: str = "valid_test.txt"
    num_workers: int = 3


@dataclass
class TrainConfig:
    max_tokens: int = 2 ** 13
    valid_batch_size: int = 2
    accumulate_grad_batches: int = 1
    accelerator: Optional[str] = None
    gpus: int = 1
    gradient_clip_val: float = 1.0
    max_epochs: int = 1000
    num_nodes: int = 1
    precision: str = "32"
    patience: int = 10
    mask_prob: float = 0.15
    random_token_prob: float = 0.1
    leave_unmasked_prob: float = 0.1


@dataclass
class LoggingConfig:
    wandb_project: Optional[str] = None
    log_every_n_steps: int = 50
    progress_bar_refresh_rate: int = 1
    track_grad_norm: bool = False


@dataclass
class Config:
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fast_dev_run: bool = False
    resume_from_checkpoint: Optional[str] = None
    val_check_interval: int = 1000


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="model", name="default", node=TransformerConfig)
cs.store(group="optimizer", name="default", node=OptimizerConfig)
cs.store(group="logging", name="default", node=LoggingConfig)


@hydra.main(config_name="config")
def train(cfg: Config) -> None:
    torch.set_float32_matmul_precision('high')
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    vocab = Vocab.from_esm_alphabet(alphabet)
    train_data = EncodedFastaDataset(cfg.data.train_fasta_path, vocab)
    train_data = RandomCropDataset(train_data, cfg.model.max_seqlen)
    train_data = MaskedTokenWrapperDataset(
        train_data,
        cfg.train.mask_prob,
        cfg.train.random_token_prob,
        cfg.train.random_token_prob,
    )
    train_sampler = BatchBySequenceLength(train_data, cfg.train.max_tokens, shuffle=True)
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=train_data.collater,
    )

    if cfg.data.valid_fasta_path:
        valid_sequence_data = EncodedFastaDataset(cfg.data.valid_fasta_path, vocab)
        valid_sequence_data = RandomCropDataset(valid_sequence_data, cfg.model.max_seqlen)
        valid_sequence_data = MaskedTokenWrapperDataset(
            valid_sequence_data,
            cfg.train.mask_prob,
            cfg.train.random_token_prob,
            cfg.train.random_token_prob,
        )
        valid_sequence_sampler = BatchBySequenceLength(valid_sequence_data, cfg.train.max_tokens, shuffle=True)
        valid_sequence_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            valid_sequence_data,
            batch_sampler=valid_sequence_sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=train_data.collater,
        )
    else:
        valid_sequence_loader = None


    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_train_split) as f:
        train_pdbs = f.read().splitlines()

    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_valid_split) as f:
        valid_pdbs = f.read().splitlines()

    trrosetta_train_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        vocab,
        split_files=train_pdbs,
        max_seqs_per_msa=1,
    )

    trrosetta_valid_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        vocab,
        split_files=valid_pdbs,
        max_seqs_per_msa=1,
    )

    valid_contact_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        trrosetta_valid_data,
        batch_size=cfg.train.valid_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=trrosetta_valid_data.collater,
    )

    model = ESM1b(
        vocab=vocab,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer,
        contact_train_data=trrosetta_train_data,
    )

    # Requires wandb to be installed
    logger = (
        pl.loggers.WandbLogger(project=cfg.logging.wandb_project)
        if cfg.logging.wandb_project is not None
        else True
    )

    if isinstance(logger, pl.loggers.Logger):
        logger.log_hyperparams(cfg.train)  # type: ignore
        logger.log_hyperparams(cfg.model)  # type: ignore
        logger.log_hyperparams(cfg.optimizer)  # type: ignore

    class ContactModelCheckpoint(pl.callbacks.ModelCheckpoint):
        def on_validation_epoch_end(self, trainer, pl_module):
            # Only save if the monitored metric exists (i.e., contact validation ran)
            if "valid/Long Range P@L" in trainer.callback_metrics:
                super().on_validation_epoch_end(trainer, pl_module)
    
    checkpoint_callback = ContactModelCheckpoint(
    	monitor="valid/Long Range P@L",
    	mode="max",
    	dirpath=current_directory / "checkpoints",
    	filename="best-epoch{epoch:02d}-step{step}",
    	auto_insert_metric_name=False,
    	save_top_k=5,
    )
    # Add a checkpoint callback to save at end of each epoch (not just on improved metric)
    end_of_epoch_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=current_directory / "checkpoints",
        filename="epoch{epoch:02d}",
        auto_insert_metric_name=False,
        save_top_k=-1,  # keep all
        every_n_epochs=1,
        save_last=False,
        save_on_train_epoch_end=True,
    )
    # # Currently it stops too early so I had to disable it. I think it's because I was testing too often and initially the model
    # # performance looks like it is going down.
    #
    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="valid/Long Range P@L",
    #     mode="max",
    #     patience=cfg.train.patience,
    # )
    lr_logger = pl.callbacks.LearningRateMonitor()

    if valid_sequence_loader is None:
        val_loader = CombinedLoader({"P@L": valid_contact_loader}, mode="max_size_cycle")
    else:
        val_loader = CombinedLoader({"P@L": valid_contact_loader, "sequence": valid_sequence_loader}, mode="max_size_cycle")

    # See https://lightning.ai/docs/pytorch/stable/upgrade/from_1_4.html for:
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_logger, checkpoint_callback, end_of_epoch_checkpoint],
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        accelerator=cfg.train.accelerator,
        fast_dev_run=cfg.fast_dev_run,
        devices=cfg.train.gpus,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        max_epochs=cfg.train.max_epochs,
        max_steps=cfg.optimizer.max_steps,
        num_nodes=cfg.train.num_nodes,
        precision=cfg.train.precision,
        val_check_interval=cfg.val_check_interval * cfg.train.accumulate_grad_batches,
        # val_check_interval=0.2,
        strategy="ddp",
        use_distributed_sampler=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.resume_from_checkpoint)


if __name__ == "__main__":
    train()