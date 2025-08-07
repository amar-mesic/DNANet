import json
import logging
import os
from datetime import datetime
import random
from typing import Optional

import confidence
import neptune
from confidence import loadf, dumpf, Configuration
import numpy as np
from torch import seed
import torch

from DNAnet.evaluation.segmentation.allele_metrics import allele_f1_score, allele_precision, allele_recall
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        training_config: str,
        checkpoint_dir: Optional[str] = None,

):
    
    training_kwargs = load_training_config(training_config)
    log_neptune = training_kwargs.get('log_neptune', False)
    
    # Set random seeds for reproducibility
    seed = training_kwargs.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_name = training_kwargs.get('experiment_name', "Prediction model")

    if log_neptune:
        run = neptune.init_run(
            name=experiment_name,
            # key="MOD", 
            project="amar-mesic/dna-thesis", 
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTQ1Njc4MC0yOTcyLTRlMmQtYTMwMy0xOGYxZTAwMmIzZGUifQ==", # your credentials
        )


    
    # Set up output directory
    output_dir = os.path.join("output", experiment_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging to file
    log_path = prepare_output_file(os.path.join(output_dir, 'log_training.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")


    

    # Load the full config, not just dataset
    full_config = load_config(data_config, kind='data')
    split_cfg = full_config.get('split', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    # log the split configuration
    LOGGER.info(f"Split configuration: {split_cfg}")

    dataset = load_dataset(data_config)

    # Split real dataset only
    train_ratio = split_cfg['train']
    val_ratio = split_cfg['val']
    test_ratio = split_cfg['test']

    train_split_is_seq = isinstance(train_ratio, confidence.models.ConfigurationSequence) # type: ignore
    total = 0 if train_split_is_seq else train_ratio
    total += val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, but got {total}.")

    # Branch based on type of train_ratio
    if train_split_is_seq:
        # Genotype-based split
        train_genotypes = set(train_ratio)
        train_set, val_test_set = dataset.split_by_genotypes(train_genotypes)
    else:
        # Ratio-based split
        train_set, val_test_set = dataset.split(train_ratio, seed)

    val_set, test_set = val_test_set.split(val_ratio / (val_ratio + test_ratio), seed)



    # Check that all splits contain at least one item
    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        raise ValueError(
            f"Each split must contain at least one item, but got "
            f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
        )
    




    # pick the model architecture, and load in pretrained checkpoint weights if available
    model = load_model(model_config)
    if checkpoint_dir:
        model.load(checkpoint_dir)
        LOGGER.info(f"Loading previous model checkpoint from {checkpoint_dir}")
    else:
        LOGGER.info("Will start training from scratch")


    # Ensure the model has a .fit() method
    if not isinstance(model, TrainableModel):
        raise ValueError(f"Model {model} is not trainable.")

    # Update training_kwargs with validation set and log parameters
    training_kwargs.update({'validation_set': val_set})
    run['parameters'] = training_kwargs



    # Run the training loop
    LOGGER.info("Starting training...")
    try:
        model.fit(train_set, neptune_run=run, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")


    if log_neptune:
        # Log the final model performance
        test_predictions = model.predict_batch(test_set)
        run['test/predictions'] = test_predictions
        # run['test/loss'] = ...
        run['test/f1'] = allele_f1_score(test_set, test_predictions)
        # LOGGER.info(f"Test F1 score: {run['test/f1'].fetch()}")
        run['test/precision'] = allele_precision(test_set, test_predictions)
        # LOGGER.info(f"Test Precision: {run['test/precision'].fetch()}")
        run['test/recall'] = allele_recall(test_set, test_predictions)
        # LOGGER.info(f"Test Recall: {run['test/recall'].fetch()}



    # Save the trained model checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint")
    model.save(checkpoint_path)
    LOGGER.info(f"Model checkpoint saved to {checkpoint_path}")

    # Save the config files for reproducibility
    config_path = os.path.join(output_dir, 'config_used.yaml')
    complete_config = simple_dump_config(
        config_path,
        os.path.join("config", "data", data_config),
        os.path.join("config", "models", model_config),
        os.path.join("config", "training", training_config)
    )
    LOGGER.info(f"Config written to {config_path}")


    with open(config_path, "r") as f:
        config_str = f.read()
    run['config'] = config_str

    run.stop()






def simple_dump_config(
    path: str,
    data_config_path: str,
    model_config_path: str,
    training_config_path: str
):
    config = {}
    config['data'] = dict(loadf(data_config_path))
    config['model'] = dict(loadf(model_config_path))
    if training_config_path:
        config['training'] = dict(loadf(training_config_path))
    dumpf(Configuration(config), path)
    return config