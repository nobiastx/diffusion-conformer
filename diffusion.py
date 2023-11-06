import torch
import numpy as np
from typing import Dict, Tuple
from torch import Tensor
from rdkit import Chem
from tqdm import tqdm
from collections import namedtuple


"""Higher level objects for training and generation"""


class weighted_loss:
    """Weighted loss criteria
    
    Parameters
    ----------
    weights : torch.tensor
        The array of weights of shape (N,)
    """
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, truth, pred):
        """Perform calculation
        
        Parameters
        ----------
        truth : torch.tensor
            The truth tensor
        pred : torch.tensor
            The prediction

        Returns
        -------
        float
            The loss
        """
        d2 = (truth-pred)**2
        return torch.einsum('ijk,j->ijk', d2, self.weights).mean()


class trainer:
    """Training helper
    
    Parameters
    ----------
    rng : torch.Generator
        Random number generator
    learning_rate : float, default 0.001
        Learning rate for optimizer
    weight_decay : float, default 0.0
        Regularization setting for optimizer
    gamma_decay : float, default 0.999
        Decay rate for learning rate
    """
    def __init__(self, rng, learning_rate=0.001, weight_decay=0.0, gamma_decay=0.999):
        self.rng = rng
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma_decay = gamma_decay

    def train( self, model, train_set, validation_set, log, num_steps=100, rho=6, sigma_max=8, sigma_min=1E-5, num_epoch=10, weight_floor=1E-5 ):
        """Train a model
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        train_set : iterable
            Iterator over minibatches for training
        validation_set : iterable
            Iterator over minibatches for validation
        log : logger object
            The logger object for recording progress
        num_steps : int, default 100
            Number of denoising steps
        rho : float, default 6
            The rho parameter used for calculating denoising step schedule
        sigma_max : float, default 8
            Maximum denoising sigma
        sigma_min : float, default 1E-5
            Minimum denoising sigma
        num_epoch : int, default 10
            Total number of epochs to run
        weight_floor : float, default E-5
            The floor used for the loss weighting function
        """
        optimizer = torch.optim.AdamW(model.parameters(),  betas=(0.9, 0.99), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma_decay)
        
        sigma_sched = torch.concat([
            torch.tensor([0.0]),
            torch.pow(torch.linspace(sigma_min**(1/rho), sigma_max**(1/rho), steps=num_steps-1),rho)
        ]).to(model.device)

        criterion = weighted_loss(1/torch.sqrt(sigma_sched**2+weight_floor**2))

        for epoch in range(num_epoch):
            total_loss = 0
            model.train()

            #
            # Standard training, mini batches
            #
            for batch in tqdm(train_set):
                batch = batch.to(device=model.device)
                optimizer.zero_grad(set_to_none=True)
                truth = torch.clone(batch['coords']).unsqueeze(1).expand(-1,sigma_sched.shape[0],-1)
                noised = batch.copy()

                noised['coords'] = torch.normal(
                    truth,
                    sigma_sched.view(1,-1,1).expand(*truth.shape),
                    generator=self.rng,
                )

                pred = model(noised, sigma_sched)
                loss = criterion(truth,pred)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            #
            # Repeat, for validation set. No gradients and model in evaluation mode.
            #
            model.eval()
            with torch.no_grad():
                validation_loss = 0.0
                for batch in tqdm(validation_set):
                    batch = batch.to(device=model.device)
                    truth = torch.clone(batch['coords']).unsqueeze(1).expand(-1,sigma_sched.shape[0],-1)
                    noised = batch.copy()

                    noised['coords'] = torch.normal(
                        truth,
                        sigma_sched.view(1,-1,1).expand(*truth.shape),
                        generator=self.rng
                    )

                    pred = model(noised, sigma_sched)
                    validation_loss += criterion(truth,pred).item()

            scheduler.step()
            log.report(epoch, total_loss, validation_loss, optimizer)
            log.save_progress(epoch, model)

        log.save_progress(num_epoch, model)


class RDKitReporter:
    """Report generated results as RDKit molecules
    
    Parameters
    ----------
    mol : Chem.Mol
        Prototype RDKit molecule
    number : int
        Number of copies expected
    """
    def __init__(self, mol, number):
        self.result = [Chem.Mol(mol) for _ in range(number)]

    @staticmethod
    def add_one_conformer(mol, coords):
        """Add coordinates to a given RDKit molecule
        
        Parameters
        ----------
        mol : Chem.Mol
            Prototype RDKit molecule to modify
        coords : numpy.array
            The coordinate values

        Notes
        -----
        Assumes that the coords array is large enough to
        account for all atoms in the molecule.
        """
        conformer = Chem.Conformer(coords.shape[0])
        for i in range(coords.shape[0]):
            conformer.SetAtomPosition(i,coords[i,:].astype(np.double))
        mol.AddConformer(conformer, assignId=True)

    def record_coords(self, coords, iteration):
        """Record coordinates
        
        Parameters
        ----------
        coords : numpy.array
            The coordinate values for all copies of the molecule
        iteration : int
            The interation number
        """
        coords = coords.detach().to("cpu").numpy()
        for i in range(coords.shape[1]):
            self.add_one_conformer(self.result[i],coords[:,i,:])

    def contents(self):
        """Return list of updated molecules"""
        return self.result


def generator_argparse(parser):
    """Update argparse with standard generation parameters
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser object to update
    """
    parser.add_argument("-w", "--scale", "--width", type=float, default=2.5, help="Initial scale (Angstroms)")
    parser.add_argument("--noise", type=float, default=4.0, help="Level of stochastic noise to inject")
    parser.add_argument("--clamp", type=float, default=100.0, help="Hyperparameter clamp")
    parser.add_argument("--rho", type=float, default=0, help="Hyperparameter rho")
    parser.add_argument("--minnorm", type=float, default=0.0006, help="Hyperparameter minnorm")
    parser.add_argument("--alpha", type=float, default=0.0, help="Hyperparameter alpha")
    parser.add_argument("--dampen", type=float, default=0.0, help="Hyperparameter dampen")


class Generator:
    """Generation helper
    
    Parameters
    ----------
    model : models.DiffusionTransformer
        The diffusion model to be trained
    steps : int
        The number of steps used for generation
    params : object with named fields
        Values for hyperparameters

    Attributes
    ----------
    params : namedtuple
        Prototype for named hyperparameters
    """
    params = namedtuple('generation_params', ['scale','noise','clamp','rho','minnorm', 'alpha', 'dampen'])

    def __init__(self, model, steps, params, external=lambda x, sigma: 0):
        self.model = model
        self.steps = steps
        self.params = params
        self.external = external

    def generate_from_mol( self, mol, data, number=1):
        """Generate conformer for a given molecule
        
        Parameters
        ----------
        mol : Chem.Mol
            The RDKit molecule of interest
        data : tuple[str,numpy.array]
            The standard dictionary object for the molecule
        number : int, Default 1
            Number of conformers to generate

        Returns
        -------
        list[Chem.Mol]
            The resulting molecules, with all intermediate conformers
            stored. Final conformer is last.
        """
        x = {
            'atoms': torch.tensor(data['atoms'], dtype=int, device=self.model.device),
            'bonds': torch.tensor(data['bonds'], dtype=int, device=self.model.device),
            'angles': torch.tensor(data['angles'], dtype=int, device=self.model.device),
            'propers': torch.tensor(data['propers'], dtype=int, device=self.model.device),
            'impropers': torch.tensor(data['impropers'], dtype=int, device=self.model.device),
            'pairs': torch.tensor(data['pairs'], dtype=int, device=self.model.device),
            'tetras': torch.tensor(data['tetras'], dtype=int, device=self.model.device),
            'cistrans': torch.tensor(data['cistrans'], dtype=int, device=self.model.device)
        }

        return self.generate(x, number, RDKitReporter(mol, number)).contents()

    def generate(self, x: Dict[str,Tensor], number: int, reporter):
        """Generate conformer for given dictionary object

        Parameters
        ----------        
        x : tuple[str,torch.tensor]
            The standard dictionary object for the molecule, as pytorch tensors
        number : int, Default 1
            Number of conformers to generate
        reporter : Reporter object
            The object to receive conformer results

        Returns
        -------
        Reporter object
            The reporter object as provided, after results are stored

        Notes
        -----
        It is assumed that the contents of x have been placed on the same
        device as the model.
        """
        self.model.eval()

        num_nodes = len(x['atoms'])

        scratch = torch.empty((num_nodes,number,3), device=self.model.device)
        zeros = torch.zeros((1,1,1), device=self.model.device).expand(num_nodes,number,3)

        #
        # Decide on sigma steps
        #
        a = np.log(self.params.minnorm)
        t = torch.linspace(0,1,steps=self.steps-1)
        sigma_sched = torch.concat([
            torch.tensor([0.0]),
            self.params.scale*torch.exp(a - (a+self.params.rho)*t + self.params.rho*t*t)
        ]).to(self.model.device)

        #
        # Convert sigma to equivalent time. Here we are using sigma(t) = t.
        #
        times = sigma_sched/self.params.scale

        with torch.no_grad():
            encoding = self.model.encoder(x)

            coords = torch.normal(zeros,self.params.scale)
            reporter.record_coords(coords, iteration=len(times))

            def get_dx(coords, sigi):
                s = sigi.expand(coords.shape[1])
                return coords - self.model.geometry(x,encoding,s) - self.external(x,s)

            for i in range(len(times)-1,0,-1):
                #
                # Establish time interval and smear coordinates as appropriate
                #
                ti = times[i]*(1+self.params.noise)
                coords = torch.normal(coords,self.params.dampen*self.params.scale*torch.sqrt((ti**2 - times[i]**2).clamp(min=0)))
                sigi = self.params.scale*ti

                #
                # Solve
                #
                x['coords'] = coords
                dt = times[i-1] - ti
                dxdt0 = get_dx(coords,sigi)/ti
                scratch = coords + dt*dxdt0.clamp(min=-self.params.clamp,max=self.params.clamp)

                if times[i-1] > 0:
                    x['coords'] = scratch
                    dxdt1 = get_dx(scratch, sigma_sched[i-1])/times[i-1]
                    scratch = coords + 0.5*dt*(
                        (1+self.params.alpha)*dxdt0 + 
                        (1-self.params.alpha)*dxdt1.clamp(min=-self.params.clamp,max=self.params.clamp)
                    )

                scratch -= torch.mean(scratch,dim=0,keepdim=True)
                reporter.record_coords(scratch, iteration=i)
                coords = scratch

            return reporter
