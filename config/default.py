def get_config():
    r"""
	Use:
        Write down all configurations before return.
    Returns:
        A dict of parameters for trainer.
	"""
    experiment_config = {
        'sr_rate': 2,  # Super resolution scale.
        'exp_name': 'baseline',  # Experiment name. All files are saved in this folder.
        # 'save_path': '/opt/tiger/ckpt',  # Path which save all experiments.
        'hdfs_path': 'hdfs://haruna/home/byte_search_nlp_lq/user/sunqingyan/lfdata/train',
        'save_path': '../ckpt',  # Path which save all experiments.
        'check_trainer': True,  # Check the whole pipeline before training.
        'saving_interval': 5,  # Save snapshot every saving_interval epochs.
        'eval_interval': 5,  # Test model every eval_interval epochs.
    }

    training_config = {
        'model': 'MDFN',  # Model will load from this python file, which in model folder.
        'loss': 'l1',  # Loss will load form this python file, which in loss folder.
        'lr': 1e-3,  # Learning rate.
        'cyclic_lr': False,  # Cyclic learning rate is used in finetune, and max_epoch should be set as 200.
        'batch_size': 32,  # Batch size.
        'max_epoch': 200,  # Trainer will stop after max_epoch epochs.
    }

    gpu_config = {
        'use_gpu': True,  # Whether use gpu to training.
        'visible_gpu': -1,  # -1 or list. -1 means use all GPUs, or specified GPUs ids in list.
    }

    loss_config = {}  # Configurations which will be given to loss function.

    network_config = {  # Configurations which will be given to network.
        'df_size': 5,
        'sr_rate': experiment_config['sr_rate']
    }

    train_dataset_config = {
        'data_path': '../dataset/130LF',
        'view_size': 7,
        'crop_size': 32,
        'repeat_rate': training_config['batch_size'],
        'sr_rate': experiment_config['sr_rate'],
        'store_in_memory': True
    }

    test_dataset_config = {
        'data_path': '../dataset/general',
        'view_size': train_dataset_config['view_size'],
        'sr_rate': experiment_config['sr_rate'],
        'store_in_memory': True
    }

    return locals()
