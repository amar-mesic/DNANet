import json
import logging
import os
from datetime import datetime
from typing import Optional

import confidence
import neptune
from confidence import loadf, dumpf, Configuration
import numpy as np
from torch import seed
import random

import torch

from DNAnet.evaluation.segmentation.allele_metrics import allele_f1_score, allele_precision, allele_recall
from DNAnet.evaluation.segmentation.pixel_metrics import pixel_f1_score, pixel_precision, pixel_recall
from DNAnet.evaluation.visualizations import plot_lanes_overlay
from DNAnet.preprocessing.pipeline import PreprocessingPipeline
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file
from DNAnet.data.data_models.base import SimpleDataset


LOGGER = logging.getLogger('dnanet')


def run(real_data_config: str,
        synth_data_config: str,
        model_config: str,
        training_config: str,
        preprocessing_steps: Optional[PreprocessingPipeline] = None,
        checkpoint_dir: Optional[str] = None):
    
    training_kwargs = load_training_config(training_config)
    log_neptune = training_kwargs['log_neptune']

        # Set random seeds for reproducibility
    seed = training_kwargs['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_name = training_kwargs['experiment_name']

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
    split_cfg = full_config['split']

    # Load datasets
    real_dataset = load_dataset(real_data_config)
    synth_dataset = load_dataset(synth_data_config)

    # Confirm if datasets are properly scaled
    if log_neptune:
        real_data = np.stack([image.data for image in real_dataset])
        synth_data = np.stack([image.data for image in synth_dataset])
        fig = plot_lanes_overlay(real_data, synth_data, n_lanes=5, show_synth=True)
        run["visualizations/train_set_distribution"].append(fig)


    # Apply preprocessing steps if provided
    if preprocessing_steps is not None:
        LOGGER.info(f"Applying preprocessing steps: {preprocessing_steps}")
        for image in real_dataset:
            image.data = preprocessing_steps.fit_transform(image.data)
        for image in synth_dataset:
            image.data = preprocessing_steps.fit_transform(image.data)
        if log_neptune:
            run['preprocessing/pipeline'] = preprocessing_steps.to_config()

    # Split real dataset only
    train_ratio = split_cfg['train']
    val_ratio = split_cfg['val']
    test_ratio = split_cfg['test']

    train_split_is_seq = isinstance(train_ratio, confidence.models.ConfigurationSequence)
    total = 0 if train_split_is_seq else train_ratio
    total += val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, but got {total}.")

    # Branch based on type of train_ratio
    if train_split_is_seq:
        # Genotype-based split
        test_genotypes = set(train_ratio)
        val_test_set, train_set = real_dataset.split_by_genotypes(test_genotypes)
    else:
        # Ratio-based split
        train_set, val_test_set = real_dataset.split(train_ratio, seed)

    val_set, test_set = val_test_set.split(val_ratio / (val_ratio + test_ratio), seed)



    # log a warning if the train set is empty
    if len(train_set) == 0:
        LOGGER.warning("Training set is empty. No real samples to train on.")

    if len(val_set) == 0 or len(test_set) == 0:
        LOGGER.warning(
            f"Each split must contain at least one item, but got "
            f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
        )



    # Select synthetic samples for training
    ratio = training_kwargs['synth_ratio']
    n_real = len(train_set)
    n_synth_needed = int((n_real + 0.0001) * ratio)
    if len(synth_dataset) >= n_synth_needed:
        # Sample without replacement
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
    LOGGER.info(f"Training set: {len(train_set)} real + {len(synth_train_set)} synthetic = {len(combined_train_set)} total (ratio used: {actual_ratio})")
    LOGGER.info(f"Validation set: {len(val_set)} real only")
    LOGGER.info(f"Test set: {len(test_set)} real only")



    # Confirm if scaling is still valid at the end
    if log_neptune:
        train_data = np.stack([image.data for image in combined_train_set])
        val_data = np.stack([image.data for image in val_set])
        test_data = np.stack([image.data for image in test_set])
        # combine val and test data
        val_test_data = np.concatenate([val_data, test_data], axis=0)

        fig = plot_lanes_overlay(train_data, val_test_data, n_lanes=5, show_synth=True)
        run["visualizations/train_set_distribution"].append(fig)




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
        run['test/pixel_f1'] = float(f"{pixel_f1_score(test_set, test_predictions):.4g}")
        run['test/pixel_precision'] = float(f"{pixel_precision(test_set, test_predictions):.4g}")
        run['test/pixel_recall'] = float(f"{pixel_recall(test_set, test_predictions):.4g}")
        
        run['test/allele_f1'] = float(f"{allele_f1_score(test_set, test_predictions):.4g}")
        run['test/allele_precision'] = float(f"{allele_precision(test_set, test_predictions):.4g}")
        run['test/allele_recall'] = float(f"{allele_recall(test_set, test_predictions):.4g}")

    # Save the trained model checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint")
    model.save(checkpoint_path)
    LOGGER.info(f"Model checkpoint saved to {checkpoint_path}")

    # Save the config files for reproducibility
    config_path = os.path.join(output_dir, 'config_used.yaml')
    complete_config = simple_dump_config(
        config_path,
        os.path.join("config", "data", real_data_config),
        os.path.join("config", "data", synth_data_config),
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
    real_data_config_path: str,
    synth_data_config_path: str,
    model_config_path: str,
    training_config_path: str
):
    config = {}
    config['real_data'] = dict(loadf(real_data_config_path))
    config['synth_data'] = dict(loadf(synth_data_config_path))
    config['model'] = dict(loadf(model_config_path))
    config['training'] = dict(loadf(training_config_path))
    dumpf(Configuration(config), path)
    return config