import os

from numpy.random import default_rng
import numpy as np

import torch
import torch.functional as F

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.dataloading import GraphDataLoader

from src.fileio import pickle_variable, unpickle_variable



def split_dataset(dataset_size:int, train_frac:float=0.5):

    rng = default_rng()
    vals = rng.uniform(low=0.0, high=1.0, size=dataset_size)

    train_mask = np.where(vals < train_frac)
    valid_mask = np.where(vals >= train_frac) and np.where(vals < (train_frac + (1-train_frac)/2))
    test_mask = np.where(vals >= (train_frac + (1-train_frac)/2))

    return (train_mask, valid_mask, test_mask)


class BMRB(DGLDataset):
    """BMRB graph dataset implemented in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 url=None,
                 raw_dir="/home/northja/datasets/bmrb/bmrb_entries_protein/",
                 save_dir="/home/northja/datasets/bmrb/processed/",
                 force_reload=False,
                 mode="prototype",
                 verbose=False):
        super(BMRB, self).__init__(name='NMR_cs_dataset',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.mode = mode

    def download(self):
        # download raw data to local disk
        #file_path = os.path.join(self.raw_dir, self.name + '.mat')
        # download file
        #download(self.url, path=file_path)
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        
        # make bidirectional
        bonds = unpickle_variable(self.save_dir + "bondlist.pkl")
        u, v = torch.tensor(bonds[0]), torch.tensor(bonds[1])

        g = dgl.graph((u, v))
        g = dgl.to_bidirected(g)

        # node features
        _features = unpickle_variable(self.save_dir + "feature_tensor.pkl")
                
        g.ndata["feat"] = torch.tensor(_features.T[1:].T) #,
                                    #dtype=F.data_type_dict['float32'])
        
        # column labels
        """
        g.ndata["headers"] = torch.tensor([
            'chem_shift', 'x', 'y', 'z', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
            'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
            'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'C', 'H', 'N', 'O', 'S',
            'has_cs'
        ])
        """

        # training labels
        g.ndata["label"] = torch.tensor(_features.T[0].T)

        # splitting masks
        
        (train_mask, val_mask, test_mask) = split_dataset(dataset_size=len(_features), train_frac=0.5)
        #print(train_mask)
        #print(val_mask)
        #print(test_mask)
        
        g.ndata['train_mask'] = train_mask[0]
        g.ndata['val_mask'] = val_mask[0]
        g.ndata['test_mask'] = test_mask[0]
        
        # node labels
        #g.ndata['label'] = torch.tensor(labels)
        
        # node features
        #g.ndata['feat'] = torch.tensor(_preprocess_features(features),
        #                               dtype=F.data_type_dict['float32'])

        #self._num_tasks = onehot_labels.shape[1]
        #self._labels = labels

        # reorder graph to obtain better locality.
        self._g = dgl.reorder_graph(g)
        self.labels = self._g.ndata["label"]

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1


    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, "prototype" + '_dgl_graph.bin')
        save_graphs(graph_path, self._g, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, "prototype" + '_info.pkl')
        #save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        self.mode = "prototype"
        graph_path = os.path.join(self.save_path, "prototype" + '_dgl_graph.bin')
        self._g, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, "prototype" + '_info.pkl')
        #self.num_classes = load_info(info_path)['num_classes']

        dataset = self.g_.ndata["feat"]

        def _collate_fn(batch):
            # batch is a list of tuple (graph, label)
            graphs = [e[0] for e in batch]
            g = dgl.batch(graphs)
            labels = [e[1] for e in batch]
            labels = torch.stack(labels, 0)
            return g, labels

        # load dataset
        #dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
        
        # dataloader
        train_loader = GraphDataLoader(dataset[self._g.ndata["train_mask"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
        valid_loader = GraphDataLoader(dataset[self._g.ndata["val_mask"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
        test_loader = GraphDataLoader(dataset[self._g.ndata["test_mask"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, "prototype" + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, "prototype" + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
