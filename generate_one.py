#!/usr/bin/env python3
import database, diffusion, external
import torch
from rdkit import Chem
import argparse, os, warnings

# we appreciate the warning, but this is to reduce output clutter
warnings.filterwarnings('ignore', message='scatter_reduce\(\) is in beta and the API may change at any time')

def arguments():
    parser = argparse.ArgumentParser("Generate a conformer")
    parser.add_argument("smiles", type=str, help="SMILES string")
    parser.add_argument("-m","--model", type=str, default="checkpoints/qmugs.pt", help="Model checkpoint file")
    parser.add_argument("-d","--device", type=str, default="cpu", help="Processing device")
    parser.add_argument("-n","--number", type=int, default=10, help="Number conformers to generate")
    parser.add_argument("-s","--steps", type=int, default=500, help="Number of noising steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-o","--output",type=str, default="output/output.sdf", help="Output sdf file")
    parser.add_argument("-t","--trace",type=str, default=None, help="Also write a trajectory file with given prefix for each generated conformer")
    parser.add_argument("--v3", action="store_true", default=False, help="Write V3 sdf files")
    parser.add_argument("--threads", type=int, default=None, help="Specify explicit number of threads")
    parser.add_argument("--repulsion", type=float, default=None, help="Overlap repulsion strength")
    diffusion.generator_argparse(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    if not os.path.exists(args.model):
        raise Exception(f"No file found at {args.model}")

    if not args.threads is None:
        torch.set_num_threads(args.threads)

    #
    # Parse smiles and sanitize, so that chirality is recognized
    #
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    #
    # Setup random seed, if so requested
    #
    rnd = torch.Generator(device=args.device)
    if not args.seed is None:
        rnd.manual_seed(args.seed)

    #
    # Select external function, if so requested
    #
    replusion = external.pair_repulsion(args.repulsion) if args.repulsion else lambda x,sigma: 0

    #
    # Generate
    #
    model = torch.load(args.model).to(args.device)
    generator = diffusion.Generator(model, steps=args.steps, params=args, external=replusion)

    mols = generator.generate_from_mol(mol, database.example(mol), args.number)

    #
    # Write final conformation to a single output file
    #
    with Chem.SDWriter(args.output) as writer:
        if args.v3: writer.SetForceV3000(True)
        for m in mols:
            writer.write(m, confId=m.GetNumConformers()-1)

    if not args.trace is None:
        #
        # As requested, write separate output files for each solution,
        # with all intermediate conformers included
        #
        for imol,mol in enumerate(mols):
            output_path = args.trace + f'{imol:03d}.sdf'
            with Chem.SDWriter(output_path) as writer:
                if args.v3: writer.SetForceV3000(True)
                for i in range(mol.GetNumConformers()):
                    writer.write(mol, confId=i)
    

