import torch_geometric as tg
import torch
from utils.transforms import OneHot, EuclideanInformationTransform



def load_md17(dataset_name, dataset_dir, radius):
    assert dataset_name.endswith('CCSD'), 'We are only interested in CCSD'
    assert 'real_datasets' in dataset_dir

    # This adds edges between all nodes that are leq radius away - otherwise there are no edges
    radius_transform = tg.transforms.RadiusGraph(r=radius)

    one_hot_transform = OneHot(input='z', name='z', delete_original=False)
    eucliean_information_transform = EuclideanInformationTransform()
    
    # TODO check whether this is correct - seems weird that we add distance last?
    transforms = tg.transforms.Compose([radius_transform,
                                        one_hot_transform,
                                        eucliean_information_transform
                                        ])

    train = tg.datasets.MD17(dataset_dir, name=dataset_name, train=True, transform=transforms)
    test = tg.datasets.MD17(dataset_dir, name=dataset_name, train=False, transform=transforms)

    return {'train': train,
            'test': test}