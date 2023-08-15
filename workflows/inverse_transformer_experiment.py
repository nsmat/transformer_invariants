from workflows.shared_workflow_helper import MD17_Experiment
from models.se3_transformer import SE3EquivariantTransformerInverseRadiusSquared

if __name__ == '__main__':
    experiment = MD17_Experiment('standard_transformer', SE3EquivariantTransformerInverseRadiusSquared)
    experiment.train_and_evaluate_model()
