from workflows.shared_workflow_helper import MD17_Experiment
from models.se3_transformer import Se3EquivariantTransformer

if __name__ == '__main__':
    print('Successfuly Running Transformer Experiment')
    experiment = MD17_Experiment('standard_transformer', Se3EquivariantTransformer)
    experiment.train_and_evaluate_model()