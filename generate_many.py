#!/usr/bin/env python3
import database, diffusion, data, external
import torch
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import argparse, os, warnings

# we appreciate the warning, but this is to reduce output clutter
warnings.filterwarnings('ignore', message='scatter_reduce\(\) is in beta and the API may change at any time')

class Batcher:
    """Build batch of several molecules
    
    Parameters
    ----------
    max_num_atoms : int
        Maximum number of atoms to include in each batch
    device : str
        PyTorch device
    """
    def __init__(self, max_num_atoms, device):
        self.max_num_atoms = max_num_atoms
        self.mols = []
        self.num_atoms = 0
        self.collator = data.MoleculeCollator(device=device)

    def add(self,mol):
        """Attempt to add a molecule
        
        Parameters
        ----------
        mol : Chem.Mol
            Molecule to add
        
        Returns
        -------
        bool
            True if addition was successful
        """
        natom = mol.GetNumAtoms()
        if natom + self.num_atoms < self.max_num_atoms:
            self.mols.append(mol)
            self.num_atoms += natom
            return True
        return False
    
    def restart(self,mol):
        """Start a new batch with this molecule
        
        Parameters
        ----------
        mol : Chem.Mol
            Molecule to add
        """
        self.mols = [mol]
        self.num_atoms = mol.GetNumAtoms()

    def contents(self):
        """Return RDKit contents"""
        return self.mols

    def batch(self):
        """Return batch"""
        return self.collator([database.example(m) for m in self.mols])


class MultiRDKitReporter:
    """Report generated results as multiple RDKit molecules
    
    Parameters
    ----------
    ref_mols : list[Chem.Mol]
        The list of molecules being generated
    output_dir : str
        Path to the directory in which to deposit contents
    v3 : bool, default False
        Save output as V3 sdf files
    """
    def __init__(self, ref_mols, output_dir, v3=False):
        self.ref_mols = ref_mols
        self.output_dir = output_dir
        self.v3 = v3

    def record_coords(self, coords, iteration):
        """Receive coordinates from the model
        
        Parameters
        ----------
        coords : Tensor
            The coordinates
        iteration : int
            The iteration begin generated
        """
        if iteration != 1: return

        coords = coords.detach().to("cpu").numpy()

        #
        # Loop over molecules, writing each to its own file
        #
        offset = 0
        for ref in self.ref_mols:
            name = ref.GetProp("_Name")
            outfile = Chem.SDWriter(os.path.join(self.output_dir,name + ".sdf"))
            if self.v3:
                outfile.SetForceV3000(True)

            #
            # Loop over conformers
            #
            for i in range(coords.shape[1]):
                mol = Chem.Mol(ref)
                conformer = Chem.Conformer(ref.GetNumAtoms())
                for j in range(ref.GetNumAtoms()):
                    conformer.SetAtomPosition(j,coords[j+offset,i,:].astype(np.double))
                confid = mol.AddConformer(conformer, assignId=True)
                outfile.write(mol, confId=confid)

            outfile.close()
            offset += ref.GetNumAtoms()




def arguments():
    parser = argparse.ArgumentParser("Generate conformers for a series of molecules")
    parser.add_argument("file", type=str, help="Input molecule SMILES file")
    parser.add_argument("-m","--model", type=str, default="checkpoints/qmugs.pt", help="Model checkpoint file")
    parser.add_argument("-d","--device", type=str, default="cuda", help="Processing device")
    parser.add_argument("-n","--number", type=int, default=10, help="Number conformers to generate per molecule")
    parser.add_argument("-s","--steps", type=int, default=500, help="Number of noising steps")
    parser.add_argument("-b","--batch", type=int, default=50000, help="Maximum batch size, in number of atoms")
    parser.add_argument("--max", type=int, default=None, help="Process only up to the given number of molecules")
    parser.add_argument("-o","--output",type=str,default="output", help="Output directory")
    parser.add_argument("--repulsion", type=float, default=None, help="Overlap repulsion strength")
    parser.add_argument("--v3", action="store_true", default=False, help="Write V3 sdf files")
    diffusion.generator_argparse(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    if not os.path.exists(args.model):
        raise Exception(f"No file found at {args.model}")

    if not os.path.exists(args.file):
        raise Exception(f"No file found at {args.file}")
    
    os.makedirs(args.output,exist_ok=True)

    #
    # Select external function, if so requested
    #
    replusion = external.pair_repulsion(args.repulsion) if args.repulsion else lambda x,sigma: 0

    #
    # Generate
    #
    model = torch.load(args.model).to(args.device)
    generator = diffusion.Generator(model, steps=args.steps, params=args, external=replusion)

    batcher = Batcher(args.batch, args.device)
    lines = list(open(args.file))
    for count,line in enumerate(tqdm(lines)):
        if (not args.max is None) and count >= args.max:
            break
        smiles,name = line.strip().split()
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        mol.SetProp("_Name",name)
        if not batcher.add(mol):
            reporter = MultiRDKitReporter(batcher.contents(), args.output, args.v3)
            generator.generate( batcher.batch(), args.number, reporter )
            batcher.restart(mol)

    if len(batcher.contents()) > 0:
        reporter = MultiRDKitReporter(batcher.contents(), args.output, args.v3)
        generator.generate( batcher.batch(), args.number, reporter )

