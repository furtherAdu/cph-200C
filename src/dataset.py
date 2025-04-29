import tqdm
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

class FromPandasDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_dict = {k: torch.tensor([v]) for k, v in self.dataset[idx].items()}
        return sample_dict


class XYDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=128,
            num_workers=11,
            outcome_col='Y',
            input_features=[],
            dataset_name='ihdp',
            features_to_standardize=[],
            features_to_ordinalize=[],
            features_to_onehot=[],
            raw_data=None,
            n_splits=None, # int
            fold=None, # int
            **kwargs):
        super().__init__()
        assert fold in range(n_splits), f'{fold} is not in range({n_splits})'

        self.raw_data = raw_data
        self.n_samples = len(self.raw_data)
        self.dataset_name = dataset_name
        self.outcome_col = outcome_col
        self.n_outcomes = None
        self.features_to_standardize = features_to_standardize
        self.numerical_scaler = StandardScaler()
        self.features_to_ordinalize = features_to_ordinalize
        self.ordinal_encoder = OrdinalEncoder()
        self.features_to_onehot = features_to_onehot
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.n_splits = n_splits
        self.fold = fold

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_features = input_features
        self.splits = dict(zip(['train', 'val', 'test'],
                               [[], [], []]))

        # get 
        if self.n_splits is None:
            train_val_idx, test_idx = train_test_split(list(self.raw_data.index), random_state=12, test_size=.2)
        else: # maguyvered k-fold cross-validation
            skf = StratifiedKFold(n_splits=self.n_splits, random_state=12, shuffle=True)
            splits = list(skf.split(X=[0 for _ in range(self.n_samples)],
                                    y=self.raw_data[outcome_col]))
            train_val_idx, test_idx = splits[self.fold]
            train_val_idx, test_idx = self.raw_data.index[train_val_idx], self.raw_data.index[test_idx]
            
        train_idx, val_idx = train_test_split(train_val_idx, random_state=12, test_size=.2)
        self.split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    def setup(self, stage=None):
        self.n_outcomes = self.raw_data[self.outcome_col].nunique()
        
        # standardize features
        if self.features_to_standardize:
            self.numerical_scaler.fit(self.raw_data.loc[self.split_idx['train'], self.features_to_standardize])
            self.raw_data[self.features_to_standardize] = self.numerical_scaler.transform(self.raw_data[self.features_to_standardize])

        if self.features_to_ordinalize:
            self.raw_data[self.features_to_ordinalize] = self.ordinal_encoder.fit_transform(self.raw_data.loc[:, self.features_to_ordinalize])

        if self.features_to_onehot:
            self.onehot_encoder.fit(self.raw_data.loc[:, self.features_to_onehot])
            ohe_features = self.onehot_encoder.get_feature_names_out()
            self.raw_data = pd.concat([self.raw_data.drop(columns=self.features_to_onehot, axis=1),
                                       pd.DataFrame(self.onehot_encoder.transform(self.raw_data.loc[:, self.features_to_onehot]),
                                                    columns=ohe_features, index=self.raw_data.index)],
                                      axis=1)
            
            self.input_features = list((set(self.input_features) - set(self.features_to_onehot)).union(set(ohe_features)))
            
        # setup assumes data is one-hot encoded
        for idx, row in tqdm.tqdm(self.raw_data.iterrows(), desc=f'Processing {self.dataset_name} Dataset'):
            split = [k for k, v in self.split_idx.items() if idx in v][0]

            # add input features
            input_features = {'X': [row[k] for k in self.input_features]} if self.input_features else {}

            # add treatment and outcome info
            Y_features = {'Y': row[self.outcome_col]}

            # create sample
            sample = {**Y_features, **input_features}

            # add it to the split
            self.splits[split].append(sample)
        
        dataset_kwargs = {}

        if stage == 'fit':
            self.train = FromPandasDataset(self.splits['train'], **dataset_kwargs)
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'validate':
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'test':
            self.test = FromPandasDataset(self.splits['test'], **dataset_kwargs)
        if stage == 'predict':
            self.predict = FromPandasDataset(self.splits['test'], **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val.__len__(), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test.__len__(), num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.predict.__len__(), num_workers=1)