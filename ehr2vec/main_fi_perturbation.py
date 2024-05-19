"""
Compute feature importance for 'concept' features using perturbation method.
Using the five folds from the cross-validation.
"""
from datetime import datetime
import os
from os.path import abspath, dirname, join, split
from typing import Dict
import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.logger import log_config
from ehr2vec.common.loader import load_and_select_splits
from ehr2vec.common.setup import (get_args, setup_logger,
                                  update_test_cfg_with_pt_ft_cfgs, fix_tmp_prefixes_for_azure_paths,
                                  initialize_configuration_finetune)
from ehr2vec.common.utils import Data, compute_number_of_warmup_steps
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.common.saver import Saver
from ehr2vec.evaluation.utils import check_data_for_overlap
from ehr2vec.feature_importance.perturb import PerturbationModel
from ehr2vec.trainer.trainer import EHRTrainer
import matplotlib.pyplot as plt

CONFIG_NAME = 'finetune_feature_importance.yaml'
BLOBSTORE='CINF'
DEAFAULT_VAL_SPLIT = 0.2

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune_fold(cfg, train_data:Data, val_data:Data, 
                fold:int, finetune_folder:str, fi_folder:str, 
                run, test_data: Data=None, )->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    save_fold_folder = join(fi_folder, f'fold_{fold}')
    if 'scheduler' in cfg:
        logger.info('Computing number of warmup steps')
        compute_number_of_warmup_steps(cfg, len(train_data))
    os.makedirs(save_fold_folder, exist_ok=True)
    os.makedirs(join(save_fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving patient numbers")
    logger.info("Saving pids")
    torch.save(train_data.pids, join(save_fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(save_fold_folder, 'val_pids.pt'))
    if len(test_data) > 0:
        torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))
    Saver(fi_folder).save_patient_nums(train_data, val_data, folder=fold_folder)

    # initialize datasets
    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes) if len(test_data) > 0 else None

    # load BEHRT model
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute test scores')
    finetuned_model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    # initialize perturbation model
    perturbation_model = PerturbationModel(finetuned_model, cfg.model)
    assert len(train_data.vocabulary)==perturbation_model.noise_simulator.sigmas_embedding.weight.shape[0], f"Vocabulary size {len(train_data.vocabulary)} does not match sigmas size {perturbation_model.noise_simulator.sigmas_embedding.weight.shape[0]}"
    modelmanager.model_path = None # to initialize training components form scratch
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(
        perturbation_model, train_dataset)

    trainer = EHRTrainer( 
        model=perturbation_model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        test_dataset=None, # test only after training
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        run_folder=save_fold_folder,
    )
    trainer.train()
    trainer.test_dataset = test_dataset
    trainer._evaluate(checkpoint['epoch'], mode='test')
    # save sigmas from the model
    perturbation_model.save_sigmas(join(fi_folder, f'sigmas_fold_{fold}.pt'))
    log_most_important_features(perturbation_model, train_data.vocabulary)

def log_most_important_features(perturbation_model, vocabulary):
    sigmas = perturbation_model.noise_simulator.sigmas_embedding.weight.flatten().cpu().detach().numpy()
    feature_importance = 1/(sigmas+1e-9)
    inv_vocab = {v: k for k, v in vocabulary.items()}
    feature_importance_dic = {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}
    sorted_features = sorted(feature_importance_dic.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:10]
    logger.info("Biggest Feature Importances")
    logger.info(sorted_features)

def _limit_patients(indices_or_pids: list, split: str)->list:
    if f'number_of_{split}_patients' in cfg.data:
        number_of_patients = cfg.data.get(f'number_of_{split}_patients')
        if len(indices_or_pids) >= number_of_patients:
            indices_or_pids = indices_or_pids[:number_of_patients]
            logger.info(f"Number of {split} patients is limited to {number_of_patients}")
        else:
            raise ValueError(f"Number of train patients is {len(indices_or_pids)}, but should be at least {number_of_patients}")
    return indices_or_pids

def cv_loop_predefined_splits(
        data: Data, 
        predefined_splits_dir: str, 
        finetune_folder:str,
        fi_folder: str,
        run,
        test_data: Data,
        )->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        train_data, val_data = load_and_select_splits(fold_dir, data)
        train_pids = _limit_patients(train_data.pids, 'train')
        val_pids = _limit_patients(val_data.pids, 'val')
        if len(train_pids)<len(train_data.pids):
            train_data = data.select_data_subset_by_pids(train_pids, mode='train')
        if len(val_pids)<len(val_data.pids):
            val_data = data.select_data_subset_by_pids(val_pids, mode='val')
        check_data_for_overlap(train_data, val_data, test_data)
        finetune_fold(cfg=cfg, 
                      train_data=train_data, 
                      val_data=val_data, 
                      fold=fold, 
                      finetune_folder=finetune_folder, 
                      fi_folder=fi_folder, run=run, 
                      test_data=test_data)
        
    return N_SPLITS


def prepare_and_load_data():
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)
    date = datetime.now().strftime("%Y%m%d-%H%M")
    fi_folder = join(cfg.paths.output_path, f'feature_importance_perturb_{date}')
    os.makedirs(fi_folder, exist_ok=True)

    finetune_folder = cfg.paths.get("model_path")
    logger = setup_logger(fi_folder, 'feature_importance.log')
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    cfg = update_test_cfg_with_pt_ft_cfgs(cfg, finetune_folder)
    cfg = fix_tmp_prefixes_for_azure_paths(cfg, azure_context)
    
    cfg.save_to_yaml(join(finetune_folder, 'feature_importance_config.yaml'))
   
    log_config(cfg, logger)
    cfg.paths.run_name = split(fi_folder)[-1]

    if not cfg.data.get('preprocess', False):
        logger.info(f"Load processed test data from {cfg.paths.model_path}")
        data = Data.load_from_directory(cfg.paths.model_path, mode='')
    else:
        raise ValueError("Not implemented yet. Just use preprocessed data.")
        #dataset_preparer = DatasetPreparer(cfg)
        #data = dataset_preparer.prepare_finetune_data() 
    return data, mount_context, cfg, run, logger, fi_folder

def average_sigmas(fi_folder:str, n_splits:int)->torch.Tensor:
    """Average sigmas from all folds. Save to sigmas_average.pt"""
    sigmas = []
    for fold in range(1, n_splits+1):
        sigmas_tensor = torch.load(join(fi_folder, f'sigmas_fold_{fold}.pt'))
        sigmas.append(sigmas_tensor)
    sigmas = torch.stack(sigmas).mean(dim=0)
    torch.save(sigmas, join(fi_folder, 'sigmas_average.pt'))
    return sigmas

def plot_most_important_features(vocabulary: Dict[str, int], sigmas:torch.Tensor, folder:str, n_feats=20)->None:
    """Save feature importance plot"""
    inv_vocab = {v: k for k, v in vocabulary.items()}
    sigmas = sigmas.cpu().detach().numpy()
    feature_importance = 1/(sigmas+1e-9)
    # use indices to map back to vocabulary
    feature_importance_dic = {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}
    _, ax = plt.subplots(figsize=(10, 10))
    sorted_features = sorted(feature_importance_dic.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:n_feats]
    features, importances = zip(*sorted_features)
    ax.barh(features, importances)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Perturbation-based Feature Importance')
    plt.savefig(join(folder, 'feature_importance.png'))

if __name__ == '__main__':
    data, mount_context, cfg, run, logger, fi_folder = prepare_and_load_data()
    test_data = Data.load_from_directory(cfg.paths.model_path, mode='test')
    n_splits = cv_loop_predefined_splits(data, 
                            predefined_splits_dir=cfg.paths.model_path, 
                            finetune_folder=cfg.paths.model_path,
                            fi_folder=fi_folder,
                            run=run,
                            test_data=test_data)
    sigmas = average_sigmas(fi_folder, n_splits)
    plot_most_important_features(data.vocabulary, sigmas, fi_folder)
    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, fix_tmp_prefixes_for_azure_paths(cfg.paths.model_path)))
        mount_context.stop()

    logger.info('Done')
