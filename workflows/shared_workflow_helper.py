from workflows.md17_pl_module import MD17TransformerTask
import torch_geometric as tg
from utils.load_md17 import load_md17
import datetime as dt

import pytorch_lightning as pl

class MD17_Experiment:
    def __init__(self, experiment_label: str, subclass, **  model_constructor_kwargs):
        self.experiment_label = experiment_label

        # Shared Configuration For Models
        self.num_channels = 16,
        self.l_max = 3,
        self.num_features = 9,  # Derived from dataset
        self.num_attention_layers = 4,
        self.num_attention_heads = 4,
        self.radial_network_hidden_units = 32

        self.model = subclass.construct_from_number_of_channels_and_lmax(
            num_channels=self.num_channels,
            l_max=self.l_max,
            num_features=self.num_features,
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            radial_network_hidden_units=self.radial_network_hidden_units,
            **model_constructor_kwargs
        )

        # Shared Configuration for Data
        self.radius = 2
        self.pl_module = MD17TransformerTask
        self.batch_size = 64

        self.task = MD17TransformerTask(self.model)

    def fetch_data(self):
        data = load_md17(dataset_name='aspirin CCSD', dataset_dir='../real_datasets', radius=self.radius)

        train_dataloader = tg.loader.DataLoader(data['train'], batch_size=self.batch_size, shuffle=True)
        validation_dataloader = tg.loader.DataLoader(data['validation'], batch_size=self.batch_size)
        test_dataloader = tg.loader.DataLoader(data['test'], batch_size=self.batch_size)

        return train_dataloader, validation_dataloader, test_dataloader
    def train_and_evaluate_model(self):
        train_dataloader, validation_dataloader, test_dataloader = self.fetch_data()

        timestamp = dt.datetime.now()
        timestamp = dt.datetime.strftime(timestamp, '%Y-%m-%d_%H-%M')

        # Logs are saved to os.path.join(save_dir, name, version).
        logger = pl.loggers.CSVLogger(name=self.experiment_label, version=timestamp, save_dir='lightning_logs')
        trainer = pl.Trainer(max_epochs=1000, logger=logger)
        trainer.fit(self.task, train_dataloader, validation_dataloader)

