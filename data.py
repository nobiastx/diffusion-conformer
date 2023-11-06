import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict

"""Data loaders and related tools"""


class Conformers(Dataset):
    """A predefined dataset of conformers
    
    Parameters
    ----------
    dbase : database.Storage
        The data storage object
    label : str
        The name of the desired dataset
    transform : callable, default None
        Transform to apply to each object in the dataset
    """
    def __init__(self, dbase, label):
        self.dbase = dbase
        self.ids = dbase.conformers(label)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):
        return self.dbase.conformer(self.ids[idx])

    def sizes(self):
        # This is a little expensive...
        return [self.dbase.number_atoms(i) for i in self.ids]


class MoleculeEvenSampler(Sampler):
    """A psuedo-random sampler with more consistent memory load

    Parameters
    ----------
    dataset : Dataset
        The dataset of interest. Must implement method `sizes`.
    generator : Generator, default None
        Random number generator

    Notes
    -----
    Attempts to deliver molecule graphs in an order that provides
    relatively even memory load by avoiding grouping larger
    graphs together.
    """
    def __init__(self, dataset, seed=None):
        generator = np.random.default_rng(seed)
        sizes = np.array(dataset.sizes())

        #
        # We can take a few approaches. Here we pair the entries
        # by decreasing and increasing size, but in random order.
        #
        bysize = np.argsort(sizes)
        random = generator.permutation(len(sizes)//2)
        pairs = np.column_stack([
            bysize[random],
            bysize[len(sizes)-random-1]
        ])

        #
        # Make sure we can deal with odd lengths.
        #
        parts = [pairs.flatten()]
        if 2*len(random) < len(sizes):
            parts.append(bysize[len(random):len(random)+1])

        self.order = np.concatenate(parts)

    def __iter__(self):
        yield from self.order

    def __len__(self):
        return len(self.order)


class BatchDictionary(dict):
    """Simple dictionary wrapper with some pytorch semantics"""

    def to(self, *args, **kwargs):
        """Move all tensors to given device"""
        def process(item):
            return item.to(*args, **kwargs) if torch.is_tensor(item) else item

        return BatchDictionary({k:process(v) for k,v in self.items()})


class MoleculeCollator:
    """Batch collator function
    
    Parameters
    ----------
    device : str
        The device in which to save the batch data
    include_pairs : bool, default False
        Include pair data
    """
    def  __init__(self, device):
        self.device = device

    def __call__(self,batch):
        """Combine a list of graph objects into one graph
        
        Parameters
        ----------
        batch : iterable
            An interable over the separate objects to combine

        Returns
        -------
        BatchDictionary
            Combined graph object
        """
        #
        # We can treat a batch of molecules as one large molecule
        # by just adding appropriate offsets to atom indices.
        #
        # We tack on non-tensor objects plus the number of atoms
        # as ordinary python objects, for reference
        #
        ids = defaultdict(list)
        natoms = []
        atoms = []
        coords = []
        bonds = []
        angles = []
        propers = []
        pairs = []
        tetras = []
        cistrans = []
        offset = 0
        for b in batch:
            for k,v in b.items():
                if not isinstance(v, np.ndarray):
                    ids[k].append(v)
            natoms.append(len(b['atoms']))
            atoms.append(b['atoms'])
            coords.append(b.get('coords',[]))
            bonds.append(b['bonds'] + offset)
            angles.append(b['angles'] + np.array([offset,offset,offset,0],dtype=int))
            propers.append(b['propers'] + np.array([offset,offset,offset,offset,0],dtype=int))
            pairs.append(b['pairs'] + offset)
            tetras.append(b['tetras'] + np.array([offset,offset,offset,offset,0],dtype=int))
            cistrans.append(b['cistrans'] + np.array([offset,offset,offset,offset,0],dtype=int))
            offset += len(b['atoms'])

        return BatchDictionary(ids | {
            'natoms':   natoms,
            'atoms':    torch.tensor(np.concatenate(atoms   )).to(self.device,dtype=int),
            'coords':   torch.tensor(np.concatenate(coords  )).to(self.device,dtype=torch.float32),
            'bonds':    torch.tensor(np.concatenate(bonds   )).to(self.device,dtype=int),
            'angles':   torch.tensor(np.concatenate(angles  )).to(self.device,dtype=int),
            'propers':  torch.tensor(np.concatenate(propers )).to(self.device,dtype=int),
            'pairs':    torch.tensor(np.concatenate(pairs   )).to(self.device,dtype=int),
            'tetras':   torch.tensor(np.concatenate(tetras  )).to(self.device,dtype=int),
            'cistrans': torch.tensor(np.concatenate(cistrans)).to(self.device,dtype=int)
        })


class MoleculeDataLoader(DataLoader):
    """Compatible pytorch DataLoader object"""
    def __init__(self, dataset, device='cpu', **kwargs):
        super().__init__(
            dataset,
            collate_fn=MoleculeCollator(device=device),
            **kwargs,
        )


