# experiment_setup.py
import logging
import os
import random
from re import split
import confidence
from confidence import loadf, dumpf, Configuration
import numpy as np
import torch
from datetime import datetime
from DNAnet.data.data_models.base import InMemoryDataset, SimpleDataset
from DNAnet.data.data_models.custom_hid_dataset import CustomHIDDataset
from DNAnet.evaluation.visualizations import plot_lanes_overlay
from DNAnet.models.base import TrainableModel
from DNAnet.models.segmentation.trainable_unet import DNANet_UNet
from config_io import load_model
from utils import add_file_handler_to_logger, prepare_output_file

LOGGER = logging.getLogger('dnanet')

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_dir(experiment_name):
    output_dir = os.path.join("output", experiment_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(output_dir, logger):
    log_path = prepare_output_file(os.path.join(output_dir, 'log_training.txt'))
    add_file_handler_to_logger(logger, path=log_path)
    logger.info(f"Logs will be written to {log_path}")
    return log_path


def setup_neptune(log_neptune, training_kwargs):
    if log_neptune:
        import neptune
        run = neptune.init_run(
            name=training_kwargs["experiment_name"],
            project="amar-mesic/dna-thesis",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTQ1Njc4MC0yOTcyLTRlMmQtYTMwMy0xOGYxZTAwMmIzZGUifQ==",
        )
        run['parameters'] = training_kwargs
        return run
    return None


def split_dataset(dataset: CustomHIDDataset, split_cfg, seed, one_dataset):
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
        test_genotypes = set(train_ratio)
        val_test_set, train_set = dataset.split_by_genotypes(test_genotypes)
    else:
        # Ratio-based split
        train_set, val_test_set = dataset.split(train_ratio, seed)
        
        # log the split configuration
        LOGGER.info(f"Split configuration: {split_cfg}")

    val_set, test_set = val_test_set.split(val_ratio / (val_ratio + test_ratio), seed)

    if one_dataset:
        validate_split_one_dataset(train_set, val_set, test_set)
    else:
        validate_split_two_datasets(train_set, val_set, test_set)
    
    return train_set, val_set, test_set


def validate_split_one_dataset(train_set, val_set, test_set):
    # Check that all splits contain at least one item
    if len(train_set) == 0:
        raise ValueError(
            f"Train set must contain at least one item, but got "
            f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
        )
    
    # Combine real and synthetic for training
    LOGGER.info(f"Training set: {len(train_set)}")
    LOGGER.info(f"Validation set: {len(val_set)}")
    LOGGER.info(f"Test set: {len(test_set)}")


def validate_split_two_datasets(train_set, val_set, test_set):
    # log a warning if the train set is empty
    if len(train_set) == 0:
        LOGGER.warning("Training set is empty. No real samples to train on.")

    if len(val_set) == 0 or len(test_set) == 0:
        LOGGER.warning(
            f"Each split should have items, but got "
            f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
        )


def add_auxilliary_data_to_train_set(train_set: InMemoryDataset, val_set, test_set, synth_dataset: InMemoryDataset, ratio: float):
    n_real = len(train_set)
    n_synth_needed = int((n_real + 0.0001) * ratio)
    if len(synth_dataset) >= n_synth_needed:
        # Sample without replacement
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

    return SimpleDataset(combined_train_set)



def overlay_tvt_datasets(train_set, val_set, test_set, run):
    train_data = [] if len(train_set) == 0 else np.stack([image.data for image in train_set])
    val_data = [] if len(val_set) == 0 else np.stack([image.data for image in val_set])
    test_data = [] if len(test_set) == 0 else np.stack([image.data for image in test_set])
    # combine val and test data
    val_test_data = np.concatenate([val_data, test_data], axis=0)

    fig = plot_lanes_overlay(train_data, val_test_data, n_lanes=5, show_synth=True)
    run["visualizations/train_set_distribution"].append(fig)




def load_and_pretrain_model(model_config, checkpoint_dir):
    model: DNANet_UNet = load_model(model_config)
    if checkpoint_dir:
        model.load(checkpoint_dir)
        LOGGER.info(f"Loading previous model checkpoint from {checkpoint_dir}")
    else:
        LOGGER.info("Will start training from scratch")

    # Ensure the model has a .fit() method
    if not isinstance(model, TrainableModel):
        raise ValueError(f"Model {model} is not trainable.")
    
    return model




def save_model_config(output_dir, real_data_config, model_config, training_config, synth_data_config = None):
    # Save the config files for reproducibility
    config_path = os.path.join(output_dir, 'config_used.yaml')

    simple_dump_config(
        config_path,
        os.path.join("config", "data", real_data_config),
        os.path.join("config", "data", synth_data_config) if synth_data_config is not None else "",
        os.path.join("config", "models", model_config),
        os.path.join("config", "training", training_config)
    )
    LOGGER.info(f"Config written to {config_path}")

    with open(config_path, "r") as f:
        config_str = f.read()

    return config_str
    




def simple_dump_config(
    path: str,
    real_data_config_path: str,
    synth_data_config_path: str,
    model_config_path: str,
    training_config_path: str
):
    config = {}
    config['real_data'] = dict(loadf(real_data_config_path))
    if synth_data_config_path:
        config['synth_data'] = dict(loadf(synth_data_config_path))
    config['model'] = dict(loadf(model_config_path))
    config['training'] = dict(loadf(training_config_path))
    dumpf(Configuration(config), path)
    return config