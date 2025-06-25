import logging
import os
from datetime import datetime
from typing import Optional

import neptune
from confidence import loadf, dumpf, Configuration

from DNAnet.evaluation.segmentation.allele_metrics import allele_f1_score, allele_precision, allele_recall
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        training_config: str,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
):
    run = neptune.init_run(
        name=experiment_name or "Prediction model",
        # name='bruh',
        # key="MOD", 
        project="amar-mesic/dna-thesis", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTQ1Njc4MC0yOTcyLTRlMmQtYTMwMy0xOGYxZTAwMmIzZGUifQ==", # your credentials
    )


    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging to file
    log_path = prepare_output_file(os.path.join(output_dir, 'log_training.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")



    

    # Load the full config, not just dataset
    full_config = load_config(data_config, kind='data')
    split_cfg = full_config.get('split', {'train': 0.8, 'val': 0.1, 'test': 0.1})

    dataset = load_dataset(data_config)



    # pick the model architecture, and load in pretrained checkpoint weights if available
    model = load_model(model_config)

    if checkpoint_dir:
        model.load(checkpoint_dir)
        LOGGER.info(f"Loading previous model checkpoint from {checkpoint_dir}")
    else:
        LOGGER.info("Will start training from scratch")





    # Use split ratios
    train_ratio = split_cfg['train']
    val_ratio = split_cfg['val']
    test_ratio = split_cfg['test']

    train_set, val_test_set = dataset.split(train_ratio, seed)
    val_set, test_set = val_test_set.split(val_ratio / (val_ratio + test_ratio), seed)





    # Ensure the model has a .fit() method
    if not isinstance(model, TrainableModel):
        raise ValueError(f"Model {model} is not trainable.")

    training_kwargs = load_training_config(training_config)
    training_kwargs.update({'validation_set': val_set})

    run['parameters'] = training_kwargs




    # Run the training loop
    LOGGER.info("Starting training...")
    try:
        model.fit(train_set, neptune_run=run, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")



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
        os.path.join("config", "training", training_config),
        split_cfg
    )
    LOGGER.info(f"Config written to {config_path}")


    run['config'] = complete_config
    run.stop()






def simple_dump_config(
    path: str,
    data_config_path: str,
    model_config_path: str,
    training_config_path: str,
    split_cfg: dict
):
    config = {}
    config['data'] = dict(loadf(data_config_path))
    config['model'] = dict(loadf(model_config_path))
    if training_config_path:
        config['training'] = dict(loadf(training_config_path))
    if split_cfg:
        config['split'] = split_cfg
    dumpf(Configuration(config), path)
    return config