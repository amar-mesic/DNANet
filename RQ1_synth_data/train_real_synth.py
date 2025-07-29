import json
import logging
import os
from datetime import datetime
from typing import Optional

import neptune
from confidence import loadf, dumpf, Configuration
from torch import seed

from DNAnet.evaluation.segmentation.allele_metrics import allele_f1_score, allele_precision, allele_recall
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file
from DNAnet.data.data_models.base import SimpleDataset


LOGGER = logging.getLogger('dnanet')


def run(real_data_config: str,
        synth_data_config: str,
        model_config: str,
        training_config: str,
        ratio: float = 100,
        checkpoint_dir: Optional[str] = None):
    
    training_kwargs = load_training_config(training_config)
    log_neptune = training_kwargs.get('log_neptune', False)
    seed = training_kwargs.get('seed', 42)
    experiment_name = training_kwargs.get('experiment_name', "Prediction model")

    if log_neptune:
        run = neptune.init_run(
            name=experiment_name,
            project="amar-mesic/dna-thesis",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTQ1Njc4MC0yOTcyLTRlMmQtYTMwMy0xOGYxZTAwMmIzZGUifQ==",
        )

    # Set up output directory
    output_dir = os.path.join("output", experiment_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging to file
    log_path = prepare_output_file(os.path.join(output_dir, 'log_training.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")

    # Load the full config, not just dataset
    full_config = load_config(real_data_config, kind='data')
    split_cfg = full_config.get('split', {'train': 0.8, 'val': 0.1, 'test': 0.1})

    # Load datasets
    real_dataset = load_dataset(real_data_config)
    synth_dataset = load_dataset(synth_data_config)

    # Split real dataset only
    train_ratio = split_cfg['train']
    val_ratio = split_cfg['val']
    test_ratio = split_cfg['test']
    total = train_ratio + val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, but got {total}.")

    train_set, val_test_set = real_dataset.split(train_ratio, seed)
    val_set, test_set = val_test_set.split(val_ratio / (val_ratio + test_ratio), seed)

    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        raise ValueError(
            f"Each split must contain at least one item, but got "
            f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
        )

    # Select synthetic samples for training
    n_real = len(train_set)
    n_synth_needed = int(n_real * ratio)
    if len(synth_dataset) >= n_synth_needed:
        # Sample without replacement
        import random
        random.seed(seed)
        synth_indices = random.sample(range(len(synth_dataset)), n_synth_needed)
        synth_train_set = [synth_dataset[i] for i in synth_indices]
        actual_ratio = ratio
    else:
        synth_train_set = list(synth_dataset)
        if n_real > 0:
            actual_ratio = len(synth_train_set) / n_real
        else:
            actual_ratio = 0
        LOGGER.warning(f"Requested ratio {ratio} could not be satisfied. Using all {len(synth_train_set)} synthetic samples for {n_real} real samples. Actual ratio: {actual_ratio}")

    # Combine real and synthetic for training
    combined_train_set = list(train_set) + synth_train_set
    LOGGER.info(f"Training set: {len(train_set)} real + {len(synth_train_set)} synthetic (ratio used: {actual_ratio})")
    LOGGER.info(f"Validation set: {len(val_set)} real only")
    LOGGER.info(f"Test set: {len(test_set)} real only")

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

    training_kwargs = load_training_config(training_config)
    training_kwargs.update({'validation_set': val_set})
    if log_neptune:
        run['parameters'] = training_kwargs

    # Run the training loop
    LOGGER.info("Starting training...")
    combined_train_dataset = SimpleDataset(combined_train_set)
    try:
        model.fit(combined_train_dataset, neptune_run=run if log_neptune else None, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")

    if log_neptune:
        # Log the final model performance
        test_predictions = model.predict_batch(test_set)
        run['test/predictions'] = test_predictions
        run['test/f1'] = allele_f1_score(test_set, test_predictions)
        run['test/precision'] = allele_precision(test_set, test_predictions)
        run['test/recall'] = allele_recall(test_set, test_predictions)

    # Save the trained model checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint")
    model.save(checkpoint_path)
    LOGGER.info(f"Model checkpoint saved to {checkpoint_path}")

    # Save the config files for reproducibility
    config_path = os.path.join(output_dir, 'config_used.yaml')
    complete_config = simple_dump_config(
        config_path,
        os.path.join("config", "data", real_data_config),
        os.path.join("config", "models", model_config),
        os.path.join("config", "training", training_config)
    )
    LOGGER.info(f"Config written to {config_path}")

    with open(config_path, "r") as f:
        config_str = f.read()
    if log_neptune:
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