#!/usr/bin/python3
import database
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os, json, pickle

"""Command line tools for building molecule sources"""


class MolGraph:
    """Analysis of molecule graph for consistency
    
    Parameters
    ----------
    mol : Chem.Mol
        The molecule of interest
    """
    def __init__(self,mol):
        self.elems = [a.GetAtomicNum() for a in mol.GetAtoms()]
        self.bonds = set()
        for bond in mol.GetBonds():
            self.bonds.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            self.bonds.add((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    def matches(self,other):
        """Check if given molecule matches
        
        Parameters
        ----------
        other : Chem.Mol
            Molecule to compare to
        
        Returns
        -------
        bool
            True if the molecule matches
        """
        return self.elems == other.elems and self.bonds == other.bonds


def valid(mols):
    """Test valididate of given list of molecules
    
    Parameters
    ----------
    mols : list[Chem.Mol]
        The list of molecules to analyze

    Returns
    -------
    bool
        True if the molecules are consistent
    """
    #
    # All molecules must have identical topology
    #
    ref = MolGraph(mols[0])
    for m in mols[1:]:
        if not ref.matches(MolGraph(m)):
            return False

    return True



#
# QMugs
#
# Isert, Clemens, Kenneth Atz, José Jiménez-Luna, and Gisbert Schneider. 
# “QMugs, Quantum Mechanical Properties of Drug-like Molecules.” 
# Scientific Data 9, no. 1 (June 7, 2022): 273. https://doi.org/10.1038/s41597-022-01390-7.
#
# Available here: https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM
#
# Disk space requirements for source (structures only): about 32G
#
# The structures in this dataset appear to be of high quality and do not
# need filtering.
#
def process_qmugs(path='data/qmugs.db', rootdir="/data/QMugs", nrows=None):
    #
    # Storage
    #
    store = database.Storage(path)

    #
    # Master csv file (large!)
    #
    dt = pd.read_csv(os.path.join(rootdir,'summary.csv'),nrows=nrows)
    for id,row in tqdm(dt.iterrows(),total=len(dt)):
        name = row['chembl_id']
        conf_name = row['conf_id']

        #
        # Let's assume conformers are named consistently
        #
        conformer_id = int(conf_name.split("_")[-1])

        #
        # Get sdf file, using assumed directory structure
        #
        sdf = Chem.SDMolSupplier(os.path.join(rootdir,'structures',name,conf_name+".sdf"), removeHs=False)
        mol = next(sdf)

        #
        # Add
        #
        store.add_mol(mol, name, conformer_id)
        if id%1000 == 999:
            store.commit()

    store.commit()
    return store

def production_qmugs():
    store = process_qmugs("data/qmugs.db")
    store.fixed_set( 'tiny', 1000 )
    store.fixed_set( 'debug', 10000 )
    store.split_set( ['train', 'validate', 'test'], [0.8,0.1,0.1] )



#
# GEOM-drugs
#
# Axelrod, Simon, and Rafael Gómez-Bombarelli. “GEOM, Energy-Annotated Molecular Conformations 
# for Property Prediction and Molecular Generation.” 
# Scientific Data 9, no. 1 (April 21, 2022): 185. https://doi.org/10.1038/s41597-022-01288-4.
#
# wget https://dataverse.harvard.edu/api/access/datafile/4327252
# tar -xf 4327252
#
# Disk space requirements for source: about 103G
#
# This dataset suffers from a small level of anomalies presumably caused by
# the chemical structure being altered during minimization. These anomalies 
# need to be removed for our use case. The strategy used here is to remove
# any molecule in its entirety if any of the corresponding conformers mismatch.
# Even atom reordering is not permitted.
#
def process_geom(path='data/geom-drugs.db', logfile="data/geom-drugs-issues.log", rootdir="/data/GEOM/rdkit_folder/drugs", nrows=None):
    #
    # Storage
    #
    store = database.Storage(path)

    #
    # Master json
    #
    contents = []
    missing = []
    with open(os.path.join(rootdir,"summary_dic.json"),"r") as jsonl:
        for line in jsonl:
            for k,v in json.loads(line).items():
                try:
                    contents.append(v['pickle_path'])
                    if nrows is not None and len(contents) >= nrows: break
                except KeyError:
                    missing.append(k)


    print(f"Total of {len(contents)} molecules, with an additional {len(missing)} with no pickle")

    log = open(logfile,"w")

    for im,path in enumerate(tqdm(contents)):
        data = pickle.load(open(os.path.join(rootdir,"..",path),"rb"))
        mols = [conf['rd_mol'] for conf in data['conformers']]

        #
        # GEOM processing has a tendency to produce artifacts.
        # If these are not removed, bad things happen.
        #
        if valid(mols):
            for ic,mol in enumerate(mols):
                store.add_mol(mol, str(im), ic)
        else:
            log.write(data['smiles']+'\n')

        if im%1000 == 999:
            store.commit()

    store.commit()
    return store


def production_geom():
    store = process_geom("data/geom-drugs.db")
    store.fixed_set( 'tiny', 1000 )
    store.fixed_set( 'debug', 10000 )
    store.split_set( ['train', 'validate', 'test'], [0.80,0.10,0.10] )


#
# The following constructs a data source from a directory
# of sdf files
#
def process_sdf(path, srcdir):
    #
    # Storage
    #
    store = database.Storage(path)

    #
    # Find files
    #
    sdfs = [r for r in os.listdir(srcdir) if r.endswith(".sdf")]

    #
    # Process
    #
    for s in tqdm(sdfs):
        sdf = Chem.SDMolSupplier(os.path.join(srcdir,s), removeHs=False)
        mol = next(sdf)
        store.add_mol(mol, mol.GetProp("_Name"), 0)

    store.commit()
    return store


if __name__ == '__main__':
    os.makedirs("data",exist_ok=True)

    production_qmugs()    
    #production_geom()
    #process_sdf('data/csd.db', 'CSD/sdf').fixed_set('all',99999)

