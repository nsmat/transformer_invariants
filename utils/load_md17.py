import torch_geometric as tg
import torch
from utils.transforms import OneHot, EuclideanInformationTransform, KCalToMeVConversion


def load_md17(dataset_name, dataset_dir, radius):
    assert dataset_name.endswith('CCSD'), 'We are only interested in CCSD'
    assert 'real_datasets' in dataset_dir

    # This adds edges between all nodes that are leq radius away - otherwise there are no edges
    radius_transform = tg.transforms.RadiusGraph(r=radius)

    one_hot_transform = OneHot(input='z',
                               name='z',
                               delete_original=False
                               )
    euclidean_information_transform = EuclideanInformationTransform()
    units_conversion = KCalToMeVConversion()

    transforms = tg.transforms.Compose([radius_transform,
                                        one_hot_transform,
                                        euclidean_information_transform,
                                        units_conversion,
                                        ])

    train = tg.datasets.MD17(dataset_dir, name=dataset_name, train=True, transform=transforms)
    test = tg.datasets.MD17(dataset_dir, name=dataset_name, train=False, transform=transforms)

    train, validation = create_validation_split(train)

    return {'train': train,
            'validation': validation,
            'test': test
            }


def create_validation_split(train_dataset, proportion=0.05, seed=123):
    generator = torch.Generator().manual_seed(seed)
    validation_length = int(len(train_dataset)*proportion)
    train_length = len(train_dataset) - validation_length
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [train_length, validation_length],
                                                                      generator=generator)

    return train_dataset, validation_dataset
