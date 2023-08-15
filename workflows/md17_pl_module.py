import torch
import pytorch_lightning as pl
import torchmetrics


class MD17TransformerTask(pl.LightningModule):
    def __init__(self, energy_model: torch.nn.Module, lr=1e-3, force_loss_weight=500):
        super().__init__()
        self.energy_model = energy_model
        self.lr = lr
        self.force_loss_weight = force_loss_weight

        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_valid_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.force_train_metric = torchmetrics.MeanAbsoluteError()
        self.force_valid_metric = torchmetrics.MeanAbsoluteError()
        self.force_test_metric = torchmetrics.MeanAbsoluteError()

    @staticmethod
    def compute_energy_normalisers(dataset):
        sum_energies = 0
        total_nodes = 0
        force_scales = 0

        for graph in dataset:
            total_nodes += graph.num_nodes
            sum_energies += graph.energy
            force_scales += torch.linalg.vector_norm(graph.force, dim=1).sum()

        mean = sum_energies / total_nodes
        std = force_scales / total_nodes

        return mean, std

    def forward(self, graph):
        graph.pos = torch.autograd.Variable(graph.pos, requires_grad=True)
        predicted_energy = self.energy_model(graph).sum(1)
        expected_outputs = torch.ones_like(predicted_energy, requires_grad=True)

        predicted_force = -1 * torch.autograd.grad(predicted_energy,
                                                   graph.pos,
                                                   grad_outputs=expected_outputs,
                                                   create_graph=True,
                                                   retain_graph=True,
                                                   # set_detect_anomaly=True
                                                   )[0]

        # predicted_energy = predicted_energy.squeeze(-1)

        return predicted_energy, predicted_force

    def energy_and_force_loss(self, graph, energy, force):
        energy_loss = torch.nn.functional.mse_loss(energy, graph.energy)
        force_loss = torch.nn.functional.mse_loss(force, graph.force)
        loss = energy_loss + self.force_loss_weight * force_loss
        return loss

    def training_step(self, graph):
        energy, force = self(graph)
        loss = self.energy_and_force_loss(graph, energy, force)
        self.energy_train_metric(energy, graph.energy)
        self.force_train_metric(force, graph.force)

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        self.log("Energy train MAE", self.energy_train_metric, prog_bar=True)
        self.log("Force train MAE", self.force_train_metric, prog_bar=True)

    @torch.inference_mode(False)
    def validation_step(self, graph, batch_idx):
        energy, force = self.forward(graph)
        self.energy_valid_metric(energy, graph.energy)
        self.force_valid_metric(force, graph.force)

    def on_validation_epoch_end(self):
        self.log("Energy valid MAE", self.energy_valid_metric, prog_bar=True)
        self.log("Force valid MAE", self.force_valid_metric, prog_bar=True)

    def test_step(self, graph, batch_idx):
        energy, force = self.forward(graph)
        self.energy_test_metric(energy, graph.energy)
        self.force_test_metric(force, graph.force)

    def on_test_epoch_end(self):
        self.log("Energy test MAE", self.energy_test_metric, prog_bar=True)
        self.log("Force test MAE", self.force_test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        num_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
