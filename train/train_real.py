import json
import logging
from math import log
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
from DNAnet.evaluation.segmentation.pixel_metrics import pixel_f1_score, pixel_precision, pixel_recall
from DNAnet.evaluation.visualizations import plot_lanes_overlay
from DNAnet.preprocessing.pipeline import PreprocessingPipeline
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from train.experiment_setup import create_output_dir, load_and_pretrain_model, overlay_tvt_datasets, save_model_config, set_random_seeds, setup_logging, setup_neptune, split_dataset
from utils import add_file_handler_to_logger, prepare_output_file
from DNAnet.models.segmentation.trainable_unet import DNANet_UNet


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        training_config: str,
        preprocessing_steps: Optional[PreprocessingPipeline] = None,
        checkpoint_dir: Optional[str] = None,

):
    
    training_kwargs = load_training_config(training_config)
    log_neptune = training_kwargs['log_neptune']
    
    # Set random seeds for reproducibility
    seed = training_kwargs.get('seed', 42)
    set_random_seeds(seed)

    run: neptune.init_run = setup_neptune(log_neptune, training_kwargs)

    # Set up output directory
    output_dir = create_output_dir(training_kwargs['experiment_name'])

    # Set up logging to file
    setup_logging(output_dir, LOGGER)


    

    # Load the full config, not just dataset
    full_config = load_config(data_config, kind='data')
    split_cfg = full_config.get('split')

    dataset = load_dataset(data_config)


    # Confirm if dataset is properly scaled
    if log_neptune:
        all_data = np.stack([image.data for image in dataset])
        fig = plot_lanes_overlay(all_data, n_lanes=5, show_synth=False)
        run["visualizations/train_set_distribution"].append(fig)


    # Apply preprocessing steps if provided
    if preprocessing_steps is not None:
        LOGGER.info(f"Applying preprocessing steps: {preprocessing_steps}")
        for image in dataset:
            image.data = preprocessing_steps.fit_transform(image.data)
        if log_neptune:
            run['preprocessing/pipeline'] = preprocessing_steps.to_config()

    
    # Split real dataset only
    train_set, val_set, test_set = split_dataset(dataset, split_cfg, seed, one_dataset=True)


    # Confirm if datasets are properly scaled
    if log_neptune:
        overlay_tvt_datasets(train_set, val_set, test_set, run)


    # pick the model architecture, and load in pretrained checkpoint weights if available
    model: DNANet_UNet = load_and_pretrain_model(model_config, checkpoint_dir)

    # Update training_kwargs with validation set
    training_kwargs.update({'validation_set': val_set})


    # Run the training loop
    LOGGER.info("Starting training...")
    try:
        model.fit(train_set, neptune_run=run, **training_kwargs)
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


    
    config_str = save_model_config(output_dir, data_config, model_config, training_config)

    if log_neptune:
        run['config'] = config_str
        run.stop()
