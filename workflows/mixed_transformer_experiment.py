from workflows.shared_workflow_helper import MD17_Experiment
from models.se3_transformer import SE3EquivariantTransformerMixedHeads

if __name__ == '__main__':
    invariant_dict = {'0': 'normal', '1': 'normal', '2': 'inverse', '3': 'inverse'}
    experiment = MD17_Experiment('standard_transformer',
                                 SE3EquivariantTransformerMixedHeads,
                                 invariant_dictionary=invariant_dict
                                 )
    experiment.train_and_evaluate_model()
