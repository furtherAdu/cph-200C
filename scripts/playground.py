import os
from itertools import product
import pandas as pd
from src.lightning_pipelines import MLPClassifier_training_pipeline

cwd = os.getcwd()
data_dir = 'data'

def main():
    # set training variables & labels
    dataset_name = 'Diabetes10Year'
    model_name = 'MLPClassifier'
    outcome_col = 'y'

    # hyperparamters
    max_epochs = 50
    patience = 5  # for early stopping
    cv = 3

    # set hyperparameter grid to search
    hp_grids = {
        'MLPClassifier': {
            'init_lr':[1e-3, 1e-4],
            'hidden_dim':[64, 128],
            'num_layers':[1,2],
        }
    }

    # logging params
    wandb_kwargs = dict(project_name='CPH_200C', 
                        wandb_entity='furtheradu', 
                        dir_path='notebooks/..',
                        offline=True)

    # init dataframe for best params
    index = pd.MultiIndex.from_product([range(cv), *hp_grids['MLPClassifier'].values()],
                                    names=['split', *hp_grids['MLPClassifier'].keys()])
    MLP_cv_results = pd.DataFrame(columns=['auc'], index=index)

    # read in data
    selected_features = pd.read_csv(os.path.join(data_dir, 'selected_features.csv'), index_col=0)
    outcome_feature = pd.read_csv(os.path.join(data_dir, 'outcome_feature.csv'), index_col=0)

    # CV grid search on MLP
    cv = 3
    max_epochs = 1
    for fold in range(cv):
        for (init_lr, hidden_dim, num_layers) in product(*hp_grids['MLPClassifier'].values()):
            
            pipeline_out = MLPClassifier_training_pipeline(
                    raw_data=pd.concat([selected_features,
                                        outcome_feature], axis=1).loc[:1000],
                    model_name=model_name,
                    outcome_col=outcome_col,
                    input_features=selected_features.columns.tolist(),
                    dataset_name=dataset_name,
                    wandb_kwargs=wandb_kwargs,
                    max_epochs=max_epochs,
                    init_lr=init_lr,
                    patience=patience,
                    fold=fold,
                    n_splits=cv,              
                )

            trainer, model, datamodule = [pipeline_out[x] for x in ['trainer', 'model', 'datamodule']]
        
            print("Testing model")
            test_pred = trainer.test(model, datamodule)[0]
            
            # unpack test prediction AUC
            test_auc = test_pred['test_auc']
            
            # update dataframe with performance
            MLP_cv_results.loc[(fold, init_lr, hidden_dim, num_layers)] = test_auc
            

if __name__ == '__main__':
    main()