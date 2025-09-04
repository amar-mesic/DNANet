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
from DNAnet.models.segmentation.trainable_unet import DNANet_UNet
from DNAnet.preprocessing.pipeline import PreprocessingPipeline
from config_io import load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
import train.train as train
from train.experiment_setup import add_auxilliary_data_to_train_set, create_output_dir, load_and_pretrain_model, overlay_tvt_datasets, save_model_config, set_random_seeds, setup_logging, setup_neptune, split_dataset
from utils import add_file_handler_to_logger, prepare_output_file
from DNAnet.data.data_models.base import SimpleDataset


LOGGER = logging.getLogger('dnanet')


def run_cross(real_data_config: str,
        synth_data_config: str,
        model_config: str,
        training_config: str,
        meta: dict,
        preprocessing_steps: Optional[PreprocessingPipeline] = None,
        checkpoint_dir: Optional[str] = None):
    
    training_kwargs = load_training_config(training_config)
    log_neptune = training_kwargs['log_neptune']

    # Set random seeds for reproducibility
    seed = training_kwargs['seed']
    set_random_seeds(seed)

    run: neptune.init_run = setup_neptune(log_neptune, training_kwargs)

    # Set up output directory
    output_dir = create_output_dir(training_kwargs['experiment_name'])

    # Set up logging to file
    setup_logging(output_dir, LOGGER)




    # Load the full config, not just dataset
    full_config = load_config(real_data_config, kind='data')
    split_cfg = full_config['split']

    # Load datasets
    real_dataset = load_dataset(real_data_config)
    synth_dataset = load_dataset(synth_data_config)

    # Confirm if datasets are properly scaled
    if log_neptune:
        # add the meta information
        for k, v in meta.items():
            run[f'meta/{k}'] = v
            
        real_data = [] if len(real_dataset) == 0 else np.stack([image.data for image in real_dataset])
        synth_data = [] if len(synth_dataset) == 0 else np.stack([image.data for image in synth_dataset])
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
    train_set, val_set, test_set = split_dataset(real_dataset, split_cfg, seed, one_dataset=False)

    # Select synthetic samples for training
    ratio = training_kwargs['synth_ratio']
    combined_train_set = add_auxilliary_data_to_train_set(train_set, val_set, test_set, synth_dataset, ratio)
    


    # Confirm if split is still valid at the end
    if log_neptune:
        overlay_tvt_datasets(combined_train_set, val_set, test_set, run)



    # pick the model architecture, and load in pretrained checkpoint weights if available
    model: DNANet_UNet = load_and_pretrain_model(model_config, checkpoint_dir)

    # Update training_kwargs with validation set
    training_kwargs.update({'validation_set': val_set})


    # Run the training loop
    LOGGER.info("Starting training...")
    try:
        model.fit(combined_train_set, neptune_run=run if log_neptune else None, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")





    # TODO: if we allow test_set size zero, we need to handle that here
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


    
    config_str = save_model_config(output_dir, real_data_config, model_config, training_config, synth_data_config)
    
    if log_neptune:
        run['config'] = config_str
        run.stop()