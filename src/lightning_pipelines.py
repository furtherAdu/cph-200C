
from src.lightning import MLPClassifierLightning
from src.lightning import get_checkpoint_callback, get_log_dir_path, get_trainer, get_logger
from src.dataset import XYDataModule

def MLPClassifier_training_pipeline(**kwargs):
    # read out kwargs
    outcome_col = kwargs.get('outcome_col')
    input_features = kwargs.get('input_features')
    dataset_name = kwargs.get('dataset_name')
    outcome_type = kwargs.get('outcome_type')
    model_name = kwargs.get('model_name')
    wandb_kwargs = kwargs.get('wandb_kwargs', {})
    raw_data = kwargs.get('raw_data')
    features_to_standardize = kwargs.get('features_to_standardize')
    features_to_ordinalize = kwargs.get('features_to_ordinalize')
    features_to_onehot = kwargs.get('features_to_onehot')
    num_classes = kwargs.get('num_classes', 2)
    hidden_dim = kwargs.get('hidden_dim', 128)
    num_layers = kwargs.get('num_layers', 1)
    init_lr = kwargs.get('init_lr', 1e-3)
    max_epochs = kwargs.get('max_epochs', 50)
    n_splits = kwargs.get('n_splits')
    fold = kwargs.get('fold')
    
    # set up data module
    datamodule = XYDataModule(outcome_col=outcome_col,
                            input_features=input_features,
                            features_to_onehot=features_to_onehot,
                            features_to_ordinalize=features_to_ordinalize,
                            features_to_standardize=features_to_standardize,
                            dataset_name=dataset_name,
                            raw_data=raw_data,
                            fold=fold,
                            n_splits=n_splits)

    # get input dim
    input_dim = len(datamodule.input_features)
    
    # set up model
    model = MLPClassifierLightning(
        input_dim=input_dim,
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_classes=num_classes, 
        init_lr=init_lr,
        outcome_type=outcome_type
        )

    # get log dir
    log_dir_path = get_log_dir_path(model_name)

    # get checkpoint callback
    checkpoint_callback = get_checkpoint_callback(model_name, log_dir_path)

    # get logger
    logger = get_logger(model_name=model_name, **wandb_kwargs)

    # get trainer
    trainer = get_trainer(model_name,
                          checkpoint_callback, 
                          logger=logger,
                          max_epochs=max_epochs)

    print("Training model")
    trainer.fit(model, datamodule)
    
    return {'trainer': trainer, 'model':model, 'datamodule': datamodule}