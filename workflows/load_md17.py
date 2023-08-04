import torch_geometric as tg
import torch

class OneHot(tg.transforms.BaseTransform):
    def __call__(self, graph):
        graph.z = torch.nn.functional.one_hot(graph.z).float()

        return graph


def load_md17(dataset_name, dataset_dir, radius):
    assert dataset_name.endswith('CCSD'), 'We are only interested in CCSD'
    assert 'real_datasets' in dataset_dir

    # This adds edges between all nodes that are leq radius away - otherwise there are no edges
    radius_transform = tg.transforms.RadiusGraph(r=radius)

    one_hot_transform = OneHot()
    distance_transform = tg.transforms.Distance()
    
    # TODO check whether this is correct - seems weird that we add distance last?
    transforms = tg.transforms.Compose([radius_transform,
                                        one_hot_transform,
                                        distance_transform
                                        ])

    train = tg.datasets.MD17(dataset_dir, name=dataset_name, train=True, transform=transforms)
    test = tg.datasets.MD17(dataset_dir, name=dataset_name, train=False, transform=transforms)

    return {'train': train,
            'test': test}