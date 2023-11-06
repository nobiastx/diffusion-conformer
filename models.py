import torch
from torch import nn, Tensor
from typing import Dict, Tuple
import math

"""PyTorch models"""

class DiffusionTransformer(nn.Module):
    """Top-level generative conformer model
    
    Parameters
    ----------
    num_token : int
        Number of atom tokens (size of atom enumeration)
    dim_embedding : int
        Size of atom embedding, also the dimension of the transformer
    dim_feed : int
        Size of intermediate layers for transformer feed-through
    num_head : int, default 5
        Number of transformer heads
    num_layer : int, default 3
        Number of transformer layers
    dropout: int, default 0.5
        Dropout applied in transformer layers
    num_aggregate_hidden
        Number of layers in geometry aggregate MLP

    Notes
    -----
    The parameter `dim_embedding` must be divisable by `num_head`.
    """
    def __init__(
        self, 
        num_token: int,
        dim_embedding: int, 
        dim_feed: int, 
        num_head: int=5,
        num_layer: int=3, 
        dropout: int=0.5, 
        num_aggregate_hidden: int=2
    ):
        super().__init__()
        self.encoder = DiffusionTransformerEncoder(
            num_token=num_token,
            dim_embedding=dim_embedding,
            dim_feed=dim_feed,
            num_head=num_head,
            num_layer=num_layer,
            dropout=dropout
        )
        self.geometry = DiffusionConformer(dim_embedding, num_hidden=num_aggregate_hidden)

    @property
    def device(self) -> torch.device:
        """Return storage device"""
        return self.geometry.device

    def forward(self, x: Dict[str,Tensor], sigma: Tensor ) -> Tensor:
        """Run the model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties, including smeared coordinates
        sigma : Tensor
            The smearing schedule, in Angstroms

        Returns
        -------
        Tensor
            The estimate of the unsmeared coordinates
        """
        return self.geometry(x, self.encoder(x), sigma)

    def probe_bond(self, x: Dict[str,Tensor], sigma: Tensor, atom0: int, atom1: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe bond response of the model for the given atoms of a given molecule
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties
        sigma : Tensor
            The smearing schedule, in Angstroms
        atom0 : int
            Index of the first atom of the bond
        atom1 : int
            Index of the second atom of the bond

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for both atoms
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0 and atom1
        """
        return self.geometry.probe_bond( self.encoder(x), sigma, atom0, atom1 )

    def probe_angle(self, x: Dict[str,Tensor], sigma: Tensor, atom0: int, atom1: int, atom2: int, ring: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe angle response of the model for the given atoms of a given molecule
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties
        sigma : Tensor
            The smearing schedule, in Angstroms
        atom0 : int
            Index of the first atom of the angle
        atom1 : int
            Index of the second atom (center) of the angle
        atom2 : int
            Index of the third atom of the angle
        ringi : int
            The associated ring binary encoding

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom2
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, and atom2
        """
        return self.geometry.probe_angle( self.encoder(x), sigma, atom0, atom1, atom2, ring )

    def probe_proper_torsion(self, x: Dict[str,Tensor], sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, theta: float) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe proper torsion response of the model for the given atoms of a given molecule
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties
        sigma : Tensor
            The smearing schedule, in Angstroms
        atom0 : int
            Index of the first (leading) atom of the torsion
        atom1 : int
            Index of the second atom of the torsion
        atom2 : int
            Index of the third atom of the torsion
        atom3 : int
            Index of the fourth (tail) atom of the torsion
        theta : float
            The assumed proper torsion angle

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, atom2, and atom3
        """
        return self.geometry.probe_proper_torsion( self.encoder(x), sigma, atom0, atom1, atom2, atom3, theta )

    def probe_tetra_chirality(self, x: Dict[str,Tensor], sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, radius: float) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe tetra chirality response of the model for the given atoms of a given molecule
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties
        sigma : Tensor
            The smearing schedule, in Angstroms
        atom0 : int
            Index of the first (central) atom of the improper torsion
        atom1 : int
            Index of the second atom of the improper torsion
        atom2 : int
            Index of the third atom of the improper torsion
        atom3 : int
            Index of the fourth (probe) atom of the improper torsion
        radius : float
            The distance from atom0 to atom3 used in the probe

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of gamma angle

        Notes
        -----
        The caller is responsible for the correct ordering of atom1, atom2, and atom3.

        Rather than attempt to sample in two dimensions, model response is sampled as a function
        of angle gamma at the fixed distance as specified.
        """
        return self.geometry.probe_tetra_chirality( self.encoder(x), sigma, atom0, atom1, atom2, atom3, radius )

    def probe_cistran(self, x: Dict[str,Tensor], sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, sign: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe cis/trans response of the model for the given atoms of a given molecule
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties
        sigma : Tensor
            The smearing schedule, in Angstroms
        atom0 : int
            Index of the first (leading) atom of the torsion
        atom1 : int
            Index of the second atom of the torsion
        atom2 : int
            Index of the third atom of the torsion
        atom3 : int
            Index of the fourth (tail) atom of the torsion
        sign : float
            The sign term distinguishing cis from trans, which should be +1 or -1 
        
        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, atom2, and atom3
        """
        return self.geometry.probe_cistran( self.encoder(x), sigma, atom0, atom1, atom2, atom3, sign )


class DiffusionTransformerEncoder(nn.Module):
    """Graph transformer model for final atom embedding
    
    Parameters
    ----------
    num_token : int
        Number of atom tokens (size of atom enumeration)
    dim_embedding : int
        Size of atom embedding, also the dimension of the transformer
    dim_feed : int
        Size of intermediate layers for transformer feed-through
    num_head : int, default 5
        Number of transformer heads
    num_layer : int, default 3
        Number of transformer layers
    dropout: int, default 0.5
        Dropout applied in transformer layers
    """
    def __init__(self, num_token: int, dim_embedding: int, dim_feed: int, num_head: int=5, num_layer: int=3, dropout: float=0.5):
        super().__init__()
        self.encoder = nn.Embedding(num_token, dim_embedding, scale_grad_by_freq=True)
        self.marker = nn.Parameter(torch.empty((2,dim_embedding)))
        nn.init.normal_(self.marker)
        self.layers = nn.ModuleList([
            GraphTransformer(dim_embedding, dim_embedding, dim_feed=dim_feed, num_head=num_head, self_term=False, dropout=dropout) for _ in range(num_layer)
        ])
        self.norm = 1.0/math.sqrt(dim_embedding)

    def forward(self, x: Dict[str,Tensor] ) -> Tensor:
        """Run the model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties, including smeared coordinates

        Returns
        -------
        Tensor
            The atoms as represented in the final embedding space
        """
        encoded = self.encoder(x['atoms'])
        #
        # The following is to mark atoms that have a special role. This is done up front,
        # but could probably be done last also.
        #
        encoded.index_add(0, x['tetras'][:,0],   self.marker[0,:].expand(x['tetras'].shape[0],-1),   alpha=0.25)
        encoded.index_add(0, x['cistrans'][:,1], self.marker[1,:].expand(x['cistrans'].shape[0],-1), alpha=0.25)
        encoded.index_add(0, x['cistrans'][:,2], self.marker[1,:].expand(x['cistrans'].shape[0],-1), alpha=0.25)

        for layer in self.layers:
            encoded = layer(encoded, x['bonds'])
        return encoded * self.norm



class AggregateMLP(nn.Module):
    """A fairly generic MLP
    
    Parameters
    ----------
    dim_start : int
        Dimension of input
    dim_hidden : int
        Dimension of hidden layers
    dim_end : int, default 1
        Dimension of output
    num_hidden : int, default 0
        Number of hidden layers
    leaky : float, default 0.001
        Value of leakiness given to the LeakyReLU activation layers
    """
    def __init__(self, dim_start: int, dim_hidden: int, dim_end: int=1, num_hidden: int=0, leaky: float=0.001):
        super().__init__()

        modules = [nn.Linear(dim_start, dim_hidden), nn.LeakyReLU(leaky)]
        for _ in range(num_hidden):
            modules.extend([nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU(leaky)])
        modules.append(nn.Linear(dim_hidden, dim_end))

        self.decode = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        for d in self.decode:
            x = d(x)
        return x


class DiffusionBondsGeometry:
    """Bond geometry helper class"""
    def __init__(self, x: Dict[str,Tensor]):
        coords = x['coords']
        self.bonds = x['bonds']

        self.dr = torch.index_select(coords,0,self.bonds[:,0]) - torch.index_select(coords,0,self.bonds[:,1])
        dl2 = torch.square(self.dr).sum(-1)
        self.dl = torch.sqrt(dl2.clamp(min=1E-12))

    def dh(self) -> Tensor:
        """Calculate unit vector pointing from atom 1 to 0"""
        return self.dr / self.dl.unsqueeze(-1)

class DiffusionBonds(nn.Module):
    """Bond length model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers
    """
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2):
        super().__init__()
        self.lengths = AggregateMLP(dim_atom_embedding*2 + 2, dim_atom_embedding, num_hidden=num_hidden, dim_end=2)

    def delta(self, geom: DiffusionBondsGeometry, encoded: Tensor, t: Tensor) -> Tensor:
        """Return predicted correction distance
        
        Parameters
        ----------
        geom : DiffusionBondGeometry
            Associated geometry object
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        """
        return self.lengths(torch.cat([
            encoded[geom.bonds[:,0]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.bonds[:,1]].unsqueeze(1).expand(-1,len(t),-1),
            t.view(1,-1,1).expand(geom.bonds.shape[0],-1,-1),
            geom.dl.unsqueeze(-1)
        ], dim=-1))

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, t: Tensor, answer: Tensor):
        """Run model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Molecule(s) properties
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        answer : Tensor
            Coordinate solution to be updated
        """
        geom = DiffusionBondsGeometry(x)
        dh = geom.dh()
        delta = self.delta(geom, encoded, t)
        answer.index_add_(0, geom.bonds[:,0], delta[:,:,0].unsqueeze(-1)*dh, alpha=-0.5)
        answer.index_add_(0, geom.bonds[:,1], delta[:,:,1].unsqueeze(-1)*dh, alpha= 0.5)

    def probe(self, encoded: Tensor, t: Tensor, atom0: int, atom1: int) -> Tuple[Tensor,Tensor]:
        """Probe response for the given atoms
        
        Parameters
        ----------
        encoded : Tensor
            Encoded atoms
        t : Tensor
            Transformed value of sigma
        atom0 : int
            Index of the first atom of the bond
        atom1 : int
            Index of the second atom of the bond

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for both atoms
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0 and atom1
        """
        nsigma = t.shape[0]
        dl = torch.linspace(0,4,steps=100).view(100,1,1).expand(100,nsigma,1)

        delta = self.lengths(torch.cat([
            encoded[atom0].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom1].view(1,1,-1).expand(100,nsigma,-1),
            t.view(1,nsigma,1).expand(100,-1,-1),
            dl
        ], dim=-1))

        return dl, delta


class DiffusionBendsGeometry:
    """Bend geometry helper class"""
    def __init__(self, x: Dict[str,Tensor]):
        coords = x['coords']
        self.angles = x['angles']

        self.dr = torch.index_select(coords,0,self.angles[:,0]) - torch.index_select(coords,0,self.angles[:,2])
        dl2 = torch.square(self.dr).sum(-1)
        self.dl = torch.sqrt(dl2.clamp(min=1E-12))

    def dh(self) -> Tensor:
        """Calculate unit vector pointing from atom 2 to 0"""
        return self.dr / self.dl.unsqueeze(-1)


class DiffusionBends(nn.Module):
    """Bend angle model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers
    num_ring_token : int, default 128
        Number of ring tokens, for ring embedding
    dim_ring_embedding : int, default 10
        Dimension of ring embedding
    """
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2, num_ring_token: int=128, dim_ring_embedding: int=10):
        super().__init__()
        self.ring_encoder = nn.Embedding(num_ring_token, dim_ring_embedding)
        self.ring_normalization = 1.0/math.sqrt(dim_ring_embedding)
        self.distances = AggregateMLP(dim_atom_embedding*3 + dim_ring_embedding + 2, dim_atom_embedding, num_hidden=num_hidden, dim_end=2)

    def delta(self, geom: DiffusionBendsGeometry, encoded: Tensor, t: Tensor) -> Tensor:
        """Return predicted correction distance
        
        Parameters
        ----------
        geom : DiffusionBondGeometry
            Associated geometry object
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        """
        return self.distances(torch.cat([
            encoded[geom.angles[:,0]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.angles[:,1]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.angles[:,2]].unsqueeze(1).expand(-1,len(t),-1),
            self.ring_encoder(geom.angles[:,3]).unsqueeze(1).expand(-1,len(t),-1)*self.ring_normalization,
            t.view(1,-1,1).expand(geom.angles.shape[0],-1,-1),
            geom.dl.unsqueeze(-1)
        ],dim=-1))

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, t: Tensor, answer: Tensor):
        """Run model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Molecule(s) properties
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        answer : Tensor
            Coordinate solution to be updated
        """
        geom = DiffusionBendsGeometry(x)
        dh = geom.dh()
        delta = self.delta(geom, encoded, t)
        answer.index_add_(0, geom.angles[:,0], delta[:,:,0].unsqueeze(-1)*dh, alpha=-0.5)
        answer.index_add_(0, geom.angles[:,2], delta[:,:,1].unsqueeze(-1)*dh, alpha= 0.5)

    def probe(self, encoded: Tensor, t: Tensor, atom0: int, atom1: int, atom2: int, ring: int) -> Tuple[Tensor,Tensor]:
        """Probe model response for the given atoms
        
        Parameters
        ----------
        encoded : Tensor
            Encoded atoms
        t : Tensor
            Transformed value of sigma
        atom0 : int
            Index of the first atom of the angle
        atom1 : int
            Index of the second atom (center) of the angle
        atom2 : int
            Index of the third atom of the angle
        ringi : int
            The associated ring binary encoding

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom2
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, and atom2
        """
        nsigma = t.shape[0]
        dl = torch.linspace(0,5.0,steps=100).view(100,1,1).expand(100,nsigma,1)
        ring = torch.tensor([ring], dtype=int)

        delta = self.distances(torch.cat([
            encoded[atom0].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom1].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom2].view(1,1,-1).expand(100,nsigma,-1),
            self.ring_encoder(ring).unsqueeze(1).expand(100,nsigma,-1)*self.ring_normalization,
            t.view(1,nsigma,1).expand(100,-1,-1),
            dl
        ], dim=-1))

        return dl, delta


class DiffusionPropersGeometry:
    """Proper torsion geometry helper class"""
    def __init__(self, x: Dict[str,Tensor]):
        coords = x['coords']
        self.propers = x['propers']

        u1 = coords[self.propers[:,1],] - coords[self.propers[:,0],]
        u2 = coords[self.propers[:,2],] - coords[self.propers[:,1],]
        u3 = coords[self.propers[:,3],] - coords[self.propers[:,2],]

        #
        # Standard torsion angle calculation
        #
        self.u1xu2 = torch.cross(u1, u2, dim=-1)
        self.u2xu3 = torch.cross(u2, u3, dim=-1)

        self.u2_norm = torch.linalg.vector_norm(u2,dim=-1,keepdim=True)
        self.u1yu2 = u1 * self.u2_norm

        self.theta = torch.atan2(
            (self.u1yu2 * self.u2xu3).sum(dim=-1),
            (self.u1xu2 * self.u2xu3).sum(dim=-1)
        )

        self.dr = torch.index_select(coords,0,self.propers[:,0]) - torch.index_select(coords,0,self.propers[:,3])
        dl2 = torch.square(self.dr).sum(-1)
        self.dl = torch.sqrt(dl2.clamp(min=1E-12))

    def dh(self):
        """Calculate unit vector pointing from atom 3 to 0"""
        return self.dr / self.dl.unsqueeze(-1)

class DiffusionPropers(nn.Module):
    """Proper torsion model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers
    """
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2):
        super().__init__()
        self.distances = AggregateMLP(dim_atom_embedding*4 + 4, dim_atom_embedding, num_hidden=num_hidden, dim_end=2)

    def delta(self, geom: DiffusionPropersGeometry, encoded: Tensor, t: Tensor) -> Tensor:
        """Return predicted correction distance
        
        Parameters
        ----------
        geom : DiffusionBondGeometry
            Associated geometry object
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        """
        return self.distances(torch.cat([
            encoded[geom.propers[:,0]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.propers[:,1]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.propers[:,2]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.propers[:,3]].unsqueeze(1).expand(-1,len(t),-1),
            t.view(1,-1,1).expand(geom.propers.shape[0],-1,-1),
            torch.sin(geom.theta).unsqueeze(-1),
            torch.cos(geom.theta).unsqueeze(-1),
            geom.dl.unsqueeze(-1),
        ],dim=2))

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, t: Tensor, answer: Tensor):
        """Run model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Molecule(s) properties
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        answer : Tensor
            Coordinate solution to be updated
        """
        geom = DiffusionPropersGeometry(x)
        dh = geom.dh()
        delta = self.delta(geom, encoded, t)
        answer.index_add_(0, geom.propers[:,0], delta[:,:,0].unsqueeze(-1)*dh, alpha=-0.5)
        answer.index_add_(0, geom.propers[:,3], delta[:,:,1].unsqueeze(-1)*dh, alpha= 0.5)

    def probe(self, encoded: Tensor, t: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, theta: float) -> Tuple[Tensor,Tensor]:
        """Probe model response for the given atoms
        
        Parameters
        ----------
        encoded : Tensor
            Encoded atoms
        t : Tensor
            Transformed value of sigma
        atom0 : int
            Index of the first (leading) atom of the torsion
        atom1 : int
            Index of the second atom of the torsion
        atom2 : int
            Index of the third atom of the torsion
        atom3 : int
            Index of the fourth (tail) atom of the torsion
        theta : float
            The assumed proper torsion angle

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, atom2, and atom3
        """
        nsigma = t.shape[0]
        dl = torch.linspace(0,8.0,steps=100).view(100,1,1).expand(100,nsigma,1)

        delta = self.distances(torch.cat([
            encoded[atom0].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom1].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom2].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom3].view(1,1,-1).expand(100,nsigma,-1),
            t.view(1,nsigma,1).expand(100,-1,-1),
            torch.sin(torch.tensor([theta])).view(1,1,1).expand(100,nsigma,1),
            torch.cos(torch.tensor([theta])).view(1,1,1).expand(100,nsigma,1),
            dl,
        ], dim=-1))

        return dl, delta



class DiffusionTetraChiralityGeometry:
    """Tetra chirality geometry helper class"""
    def __init__(self, x: Dict[str,Tensor]):
        coords = x['coords']
        tetras = x['tetras']

        #
        # To simplify calculations, we can expand to all three permutations
        #
        self.permutations = torch.concat([
            tetras,
            tetras[:,[0,3,1,2,4]],
            tetras[:,[0,2,3,1,4]],
        ],dim=0)

        v0 = coords[self.permutations[:,1],] - coords[self.permutations[:,0],]
        v1 = coords[self.permutations[:,2],] - coords[self.permutations[:,0],]
        v2 = coords[self.permutations[:,3],] - coords[self.permutations[:,0],]

        self.cross = self.permutations[:,4].view(-1,1,1)*torch.cross(v1,v2,dim=-1)
        self.cross /= torch.linalg.norm(self.cross,dim=-1,keepdim=True)
        self.sum = v1+v2
        self.sum /= torch.linalg.norm(self.sum,dim=-1,keepdim=True)

        self.out = (self.cross*v0).sum(dim=-1)
        self.along = -(self.sum*v0).sum(dim=-1)


class DiffusionTetraChirality(nn.Module):
    """Tetra chirality model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers
    """
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2):
        super().__init__()
        self.distances = AggregateMLP(dim_atom_embedding*4 + 3, dim_atom_embedding, num_hidden=num_hidden, dim_end=2)

    def delta(self, geom: DiffusionTetraChiralityGeometry, encoded: Tensor, t: Tensor) -> Tensor:
        """Return predicted correction distance
        
        Parameters
        ----------
        geom : DiffusionBondGeometry
            Associated geometry object
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        """
        return self.distances(torch.cat([
            encoded[geom.permutations[:,0]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.permutations[:,1]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.permutations[:,2]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.permutations[:,3]].unsqueeze(1).expand(-1,len(t),-1),
            t.view(1,-1,1).expand(geom.permutations.shape[0],-1,-1),
            geom.out.unsqueeze(-1)/4,
            geom.along.unsqueeze(-1)/4,
        ],dim=2))

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, t: Tensor, answer: Tensor):
        """Run model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Molecule(s) properties
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        answer : Tensor
            Coordinate solution to be updated
        """
        geom = DiffusionTetraChiralityGeometry(x)
        delta = self.delta(geom, encoded, t)
        answer.index_add_(0, geom.permutations[:,0], delta[:,:,0].unsqueeze(-1)*geom.cross, alpha=-0.25)
        answer.index_add_(0, geom.permutations[:,1], delta[:,:,1].unsqueeze(-1)*geom.cross, alpha= 0.25)

    def probe(self, encoded: Tensor, t: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, radius: float) -> Tuple[Tensor,Tensor]:
        """Probe model response for the given atoms
        
        Parameters
        ----------
        encoded : Tensor
            Encoded atoms
        t : Tensor
            Transformed value of sigma
        atom0 : int
            Index of the first (central) atom of the improper torsion
        atom1 : int
            Index of the second atom of the improper torsion
        atom2 : int
            Index of the third atom of the improper torsion
        atom3 : int
            Index of the fourth (probe) atom of the improper torsion
        radius : float
            The distance from atom0 to atom3 used in the probe

        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of gamma angle

        Notes
        -----
        The caller is responsible for the correct ordering of atom1, atom2, and atom3.

        Rather than attempt to sample in two dimensions, model response is sampled as a function
        of angle gamma at the fixed distance as specified.
        """
        nsigma = t.shape[0]
        theta = torch.linspace(-torch.pi,+torch.pi,steps=100).view(100,1,1).expand(100,nsigma,1)

        delta = self.distances(torch.cat([
            encoded[atom0].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom1].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom2].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom3].view(1,1,-1).expand(100,nsigma,-1),
            t.view(1,nsigma,1).expand(100,-1,-1),
            radius*torch.sin(theta)/4,
            radius*torch.cos(theta)/4,
        ], dim=-1))

        return theta, delta


class DiffusionCisTransGeometry:
    """Cis/trans geometry helper class"""
    def __init__(self, x: Dict[str,Tensor]):
        coords = x['coords']
        self.cistrans = x['cistrans']

        c1 = coords[self.cistrans[:,2],] + coords[self.cistrans[:,1],]
        c2 = coords[self.cistrans[:,3],] + coords[self.cistrans[:,0],]

        self.segment = (c1 - c2)*0.5
        self.dist = torch.linalg.vector_norm(self.segment, dim=-1, keepdim=True)
        self.segment /= self.dist

class DiffusionCisTrans(nn.Module):
    """Cis/trans model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers
    """
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2):
        super().__init__()
        self.distances = AggregateMLP(dim_atom_embedding*4 + 3, dim_atom_embedding, num_hidden=num_hidden, dim_end=2)

    def delta(self, geom: DiffusionCisTransGeometry, encoded: Tensor, t: Tensor) -> Tensor:
        """Return predicted correction distance
        
        Parameters
        ----------
        geom : DiffusionBondGeometry
            Associated geometry object
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        """
        return self.distances(torch.cat([
            encoded[geom.cistrans[:,0]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.cistrans[:,1]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.cistrans[:,2]].unsqueeze(1).expand(-1,len(t),-1),
            encoded[geom.cistrans[:,3]].unsqueeze(1).expand(-1,len(t),-1),
            t.view(1,-1,1).expand(geom.cistrans.shape[0],-1,-1),
            geom.dist/4,
            geom.cistrans[:,4].view(-1,1,1).expand(-1,len(t),1),
        ],dim=2))

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, t: Tensor, answer: Tensor):
        """Run model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Molecule(s) properties
        encoded : Tensor
            Atom encoding
        t : Tensor
            Transformed value of sigma
        answer : Tensor
            Coordinate solution to be updated
        """
        geom = DiffusionCisTransGeometry(x)
        delta = self.delta(geom, encoded, t)
        answer.index_add_(0, geom.cistrans[:,0], delta[:,:,0].unsqueeze(-1)*geom.segment, alpha=-0.5)
        answer.index_add_(0, geom.cistrans[:,1], delta[:,:,1].unsqueeze(-1)*geom.segment, alpha= 0.5)
        answer.index_add_(0, geom.cistrans[:,2], delta[:,:,1].unsqueeze(-1)*geom.segment, alpha= 0.5)
        answer.index_add_(0, geom.cistrans[:,3], delta[:,:,0].unsqueeze(-1)*geom.segment, alpha=-0.5)

    def probe(self, encoded: Tensor, t: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, sign: int) -> Tuple[Tensor,Tensor]:
        """Probe model response for the given atoms
        
        Parameters
        ----------
        encoded : Tensor
            Encoded atoms
        t : Tensor
            Transformed value of sigma
        atom0 : int
            Index of the first (leading) atom of the torsion
        atom1 : int
            Index of the second atom of the torsion
        atom2 : int
            Index of the third atom of the torsion
        atom3 : int
            Index of the fourth (tail) atom of the torsion
        sign : float
            The sign term distinguishing cis from trans, which should be +1 or -1 
        
        Returns
        -------
        Tuple[Tensor,Tensor,Tensor]
            The correction values for atom0 and atom3
            The input values of sigma, as provided
            The sampled values of distance

        Notes
        -----
        The caller is responsible for the correct ordering of atom0, atom1, atom2, and atom3
        """
        nsigma = t.shape[0]
        dist = torch.linspace(0,8.0,steps=100).view(100,1,1).expand(100,nsigma,1)
        signs = torch.tensor([sign]).view(1,1,1).expand(100,nsigma,1)

        delta = self.distances(torch.cat([
            encoded[atom0].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom1].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom2].view(1,1,-1).expand(100,nsigma,-1),
            encoded[atom3].view(1,1,-1).expand(100,nsigma,-1),
            t.view(1,nsigma,1).expand(100,-1,-1),
            dist/4,
            signs,
        ], dim=-1))

        return dist, delta




class DiffusionConformer(nn.Module):
    """Geometry portion of generative model
    
    Parameters
    ----------
    dim_atom_embedding : int
        Size of atom embedding
    num_hidden : int, default 2
        Number of hidden layers, applied to all components
    """    
    def __init__(self, dim_atom_embedding: int, num_hidden: int=2):
        super().__init__()
        self.bonds = DiffusionBonds(dim_atom_embedding, num_hidden)
        self.bends = DiffusionBends(dim_atom_embedding, num_hidden)
        self.propers = DiffusionPropers(dim_atom_embedding, num_hidden)
        self.tetras = DiffusionTetraChirality(dim_atom_embedding, num_hidden)
        self.cistrans = DiffusionCisTrans(dim_atom_embedding, num_hidden)

    @property
    def device(self) -> torch.device:
        # This would have been nice, but its unsupported under TorchScript
        #return next(self.parameters()).device
        return self.bends.ring_encoder.weight.device

    def sigma_parameterization(self, sigma : Tensor) -> Tensor:
        """Common parameterization of sigma"""
        return sigma/4

    def forward(self, x: Dict[str,Tensor], encoded: Tensor, sigma: Tensor) -> Tensor:
        """Run the model
        
        Parameters
        ----------
        x : Dict[str,Tensor]
            Dictionary of molecule properties, including smeared coordinates
        encoded : Tensor
            The atom encoding
        sigma : Tensor
            The smearing schedule, in Angstroms

        Returns
        -------
        Tensor
            The estimate of the unsmeared coordinates
        """
        #
        # Coords [N,T,3]
        #    N = number of atoms
        #    T = batch = (number of diffusion steps) or (number of generated copies)
        #    3 = x,y,z
        #
        t = self.sigma_parameterization(sigma)

        answer = x['coords'].clone()

        self.bonds.forward(x, encoded, t, answer)
        self.bends.forward(x, encoded, t, answer)
        self.propers.forward(x, encoded, t, answer)
        self.tetras.forward(x, encoded, t, answer)
        self.cistrans.forward(x, encoded, t, answer)
        return answer

    def probe_bond(self, encoded: Tensor, sigma: Tensor, atom0: int, atom1: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe bond response of the model for the given atoms of an encoded molecule
        
        See: :meth:`~DiffusionTransformer.probe_bond`
        """
        t = self.sigma_parameterization(sigma)
        dl, delta = self.bonds.probe(encoded, t, atom0, atom1)

        return dl.squeeze(-1), sigma, delta

    def probe_angle(self, encoded: Tensor, sigma: Tensor, atom0: int, atom1: int, atom2: int, ring: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe angle response of the model for the given atoms of an encoded molecule
        
        See: :meth:`~DiffusionTransformer.probe_angle`
        """
        t = self.sigma_parameterization(sigma)
        dl, delta = self.bends.probe(encoded, t, atom0, atom1, atom2, ring)

        return dl.squeeze(-1), sigma, delta

    def probe_proper_torsion(self, encoded: Tensor, sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, theta: float) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe proper torsion response of the model for the given atoms of an encoded molecule
        
        See: :meth:`~DiffusionTransformer.probe_proper_torsion`
        """
        t = self.sigma_parameterization(sigma)
        dl, delta = self.propers.probe(encoded, t, atom0, atom1, atom2, atom3, theta)

        return dl.squeeze(-1), sigma, delta

    def probe_tetra_chirality(self, encoded: Tensor, sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, radius: float) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe tetra chirality response of the model for the given atoms of an encoded molecule
        
        See: :meth:`~DiffusionTransformer.probe_tetra_chirality`
        """
        #
        # Proper torsion
        #
        t = self.sigma_parameterization(sigma)
        da, delta = self.tetras.probe(encoded, t, atom0, atom1, atom2, atom3, radius)

        return da.squeeze(-1), sigma, delta

    def probe_cistran(self, encoded: Tensor, sigma: Tensor, atom0: int, atom1: int, atom2: int, atom3: int, sign: int) -> Tuple[Tensor,Tensor,Tensor]:
        """Probe cis/trans response of the model for the given atoms of an encoded molecule
        
        See: :meth:`~DiffusionTransformer.probe_cistran`
        """
        #
        # Proper torsion
        #
        t = self.sigma_parameterization(sigma)
        da, delta = self.cistrans.probe(encoded, t, atom0, atom1, atom2, atom3, sign)

        return da.squeeze(-1), sigma, delta


class GraphTransformer(nn.Module):
    """Graph transformer layer employing GATv2 attention

    Parameters
    ----------
    dim_input : int
        Dimension of input and output features
    dim_embedding : int
        Dimension of embedded features
    num_head : int
        Number of heads
    dim_feed : int, default 0
        Dimension of feed hidden layers, or if 0,
        feed hidden layers are the same dimension as the embedding
    dropout : float, default 0.5
        Dropout probability, for alpha
    leaky_negative_slope : float, default 0.2
        Negative slope for leakyReLU, for epsilon
    self_term : bool, default True
        Include a self-node term. Otherwise, nodes are concatenated
        into the feed forward.

    Notes
    -----
    The value of dim_embedding should be divisible by num_head.

    Nodes values are feed directly to the feed through in parallel with the
    attention values (derived from edges).
    """
    def __init__(
        self, 
        dim_input: int,
        dim_embedding: int,
        num_head: int, 
        dim_feed: int=0,
        dropout: float = 0.5,
        self_term: bool = False,
        leaky_negative_slope: float = 0.2,
    ):
        super().__init__()

        self.self_term = self_term

        self.attention = GraphAttention(
            dim_input=dim_input, 
            dim_embedding=dim_embedding, 
            num_head=num_head, 
            dropout=dropout,
            leaky_negative_slope=leaky_negative_slope,
            self_term=self_term
        )

        dim_feed = dim_input if dim_feed == 0 else dim_feed

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embedding if self_term else dim_embedding+dim_input, dim_feed),
            nn.GELU(),
            nn.Linear(dim_feed, dim_feed),
            nn.GELU(),
            nn.Linear(dim_feed, dim_input),
        )
    
    def forward(self, nodes: Tensor, edges: Tensor ) -> Tensor:
        """Calculate forward pass
        
        Parameters
        ----------
        nodes: Tensor
            node values
        edges: Tensor
            pairs of indices indicating edges

        Returns
        -------
        Tensor
            Resulting node values

        Notes
        -----
        * Graphs are assumed to be undirected. Each graph edge should have 
        only one entry in `edges`.
        """
        attn = self.attention(nodes, edges)
        if self.self_term:
            return self.feed_forward(attn)
        else:
            return self.feed_forward(torch.cat([attn,nodes],dim=1))

        

class GraphAttention(nn.Module):
    """Graph self-attention layer (V2), with multiple heads and batch support

    Parameters
    ----------
    dim_input : int
        Dimension of input and output features
    dim_embedding : int
        Dimension of embedded features
    num_head : int
        Number of heads
    dropout : float, default 0.5
        Dropout probability, for alpha
    leaky_negative_slope : float, default 0.2
        Negative slope for leakyReLU, for epsilon
    self_term : bool, default True
        Include a self-node term

    Notes
    -----
    Requires torch 1.12 or newer.

    The value of dim_embedding should be divisible by num_head.

    GATv2 has been modified to include an optional self node term.

    Brody, Shaked, Uri Alon, and Eran Yahav. “How Attentive Are Graph Attention Networks?” 
    ArXiv:2105.14491 [Cs], January 31, 2022. http://arxiv.org/abs/2105.14491.
    """
    def __init__(
        self, 
        dim_input: int,
        dim_embedding: int,
        num_head: int, 
        dropout: float = 0.5,
        leaky_negative_slope: float = 0.2,
        self_term : bool = False
    ):
        super().__init__()

        self.num_head = num_head
        self.dim_head, remainder = divmod(dim_embedding, num_head)
        if remainder != 0:
            raise Exception("Multihead attention requires a dimension divisible by the number of heads")

        self.W_l = nn.Linear(dim_input, dim_embedding, bias=False)
        self.W_r = nn.Linear(dim_input, dim_embedding, bias=False)

        self.W_self = nn.Linear(dim_input, num_head) if self_term else None

        self.attention = nn.Linear(self.dim_head, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_negative_slope)
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self) -> torch.device:
        return self.W_l.weight.device

    def forward(self, nodes: Tensor, edges: Tensor ) -> Tensor:
        """Calculate forward pass
        
        Parameters
        ----------
        nodes: Tensor
            node values
        edges: Tensor
            pairs of indices indicating edges

        Returns
        -------
        Tensor
            Resulting node values

        Notes
        -----
        * Graphs are assumed to be undirected. Each graph edge should have 
        only one entry in parameter edges.
        """
        len_seq = nodes.shape[0]

        #
        # Build g terms by node and split by heads
        #
        ngl  = self.W_l(nodes).view(len_seq,self.num_head,self.dim_head)
        ngr  = self.W_r(nodes).view(len_seq,self.num_head,self.dim_head)

        #
        # We need two copies of all edges, to apply in both directions
        #
        index = torch.cat([edges,edges[:,[1,0]]],dim=0)

        #
        # Construct epsilon
        #
        gl = ngl[index[:,0]]
        gr = ngr[index[:,1]]
        epsilon = self.attention(self.activation(gl + gr)).squeeze(-1)
        
        #
        # If requested, append a self term. This is based conceptionally on
        # what a complete set of self edges might produce, except parameterized
        # independently.
        #
        if not self.W_self is None:
            index = torch.cat([
                index,
                torch.arange(len_seq,device=self.device).unsqueeze(-1).expand(len_seq,2)
            ],dim=0)

            epsilon = torch.cat([
                epsilon,
                self.W_self(nodes)
            ],dim=0)

            gr = torch.cat([
                gr,
                ngr
            ])

        #
        # Softmax via scatter
        #
        ind = index[:,0].unsqueeze(-1).expand(-1,epsilon.shape[1])
        smax = epsilon.scatter_reduce(src=epsilon, dim=0, index=ind, reduce="amax", include_self=False)
        smax = smax.index_select(0,index[:,0])        
        sexp = (epsilon - smax).exp()
        ssum = epsilon.scatter_reduce(src=sexp, dim=0, index=ind, reduce="sum", include_self=False)
        ssum = ssum.index_select(0,index[:,0])
        alpha = self.dropout(sexp/ssum)

        #
        # Final product, summed using scatter
        #
        byedge = gr * alpha.unsqueeze(-1)
        ind = index[:,0].view(index.shape[0],1,1).expand(-1,byedge.shape[1],byedge.shape[2])
        answer = byedge.scatter_reduce(src=byedge, dim=0, index=ind, reduce="sum", include_self=False)[:len_seq]

        return answer.view(len_seq, -1)

