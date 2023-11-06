import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
from itertools import combinations

"""Standardized data source for molecule topology and conformers"""

class AtomEnumeration:
    """Standard atom enumeration"""

    #
    # Changing the following breaks all previously created models and databases
    #
    standard_enumeration = (
        ( 1,  0, "S"   ),
        ( 5,  0, "SP2" ),
        ( 5, -1, "SP3" ),
        ( 6, -1, "SP"  ),
        ( 6,  0, "SP"  ),
        ( 6, -1, "SP2" ),
        ( 6,  0, "SP2" ),
        ( 6,  1, "SP2" ),
        ( 6, -1, "SP3" ),
        ( 6,  0, "SP3" ),
        ( 7,  0, "SP"  ),
        ( 7,  1, "SP"  ),
        ( 7, -2, "SP2" ),
        ( 7, -1, "SP2" ),
        ( 7,  0, "SP2" ),
        ( 7,  1, "SP2" ),
        ( 7, -1, "SP3" ),
        ( 7,  0, "SP3" ),
        ( 7,  1, "SP3" ),
        ( 8, -1, "SP2" ),
        ( 8,  0, "SP2" ),
        ( 8,  1, "SP2" ),
        ( 8, -1, "SP3" ),
        ( 8,  0, "SP3" ),
        ( 8,  1, "SP3" ),
        ( 9, -1, "SP3" ),
        ( 9,  0, "SP3" ),
        (14,  0, "SP"  ),
        (14,  0, "SP3" ),
        (14,  1, "SP3" ),
        (15,  1, "SP"  ),
        (15, -1, "SP2" ),
        (15,  0, "SP2" ),
        (15,  1, "SP2" ),
        (15,  0, "SP3" ),
        (15,  1, "SP3" ),
        (15,  0, "SP3D"),
        (16,  3, "S"   ),
        (16,  0, "SP"  ),
        (16,  1, "SP"  ),
        (16,  2, "SP"  ),
        (16,  3, "SP"  ),
        (16, -1, "SP2" ),
        (16,  0, "SP2" ),
        (16,  1, "SP2" ),
        (16,  2, "SP2" ),
        (16,  3, "SP2" ),
        (16, -1, "SP3" ),
        (16,  0, "SP3" ),
        (16,  1, "SP3" ),
        (16,  2, "SP3" ),
        (16,  3, "SP3" ),
        (16,  0, "SP3D"),
        (17, -1, "SP3" ),
        (17,  0, "SP3" ),
        (17,  1, "SP3" ),
        (35,  0, "SP3" ),
        (35,  1, "SP3" ),
        (53, -1, "SP3" ),
        (53,  0, "SP3" ),
        (53,  1, "SP3" ),
        (53,  2, "SP3" ),
    )
    
    def __init__(self):
        self.encoding = {e:i for i,e in enumerate(self.standard_enumeration)}

    def atom_encoding(self, atom):
        """Return enumeration for given RDKit atom
        
        Parameters
        ----------
        atom : Chem.Atom
            A RDKit atom

        Returns
        -------
        int
            The atom type index, counting from zero
        """
        atomic_number = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = str(atom.GetHybridization())

        try:
            return self.encoding[(atomic_number,formal_charge,hybridization)]
        except KeyError:
            return None
        
    def properties(self, index):
        """Return properties of given encoding
        
        Parameters
        ----------
        index : int
            Atom type index, counting from zero
        
        Returns
        -------
        (int,int,str)
            The atomic number, formal charge, and hybridization string
        """
        return self.standard_enumeration[index]
        
    def size(self):
        """Return number of atom types"""
        return len(self.encoding)


def build_ring_code(mol, a1, a2, a3):
    """Build binary encoded ring code for a bend
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule associated with given atoms
    a1 : int
        First atom of bind
    a2 : int
        Second atom of bind
    a3 : int
        Third atom of bind

    Notes
    -----
    ``
    Ring code, binary field encoding, max value 127
        bits   usage
        0      set if in ring of 3
        1-2    number of ring of size 4, up to 3
        3-4    number of ring of size 5, up to 3
        5-6    number of ring of size 6, up to 3
    
    Angle is a1 - a2 - a3, in order (a2 is central atom).
    ``
    """
    answer = 0

    #
    # Breadth first search, up to two levels.
    # Since we are only going twice, we won't bother
    # with a loop
    #
    found = set([a1,a2,a3])
    curr = set([a1,a3])
    
    #
    # First level
    #
    next = set()
    num4 = 0
    for a in curr:
        atom = mol.GetAtomWithIdx(a)
        neighs = [n.GetIdx() for n in atom.GetNeighbors()]
        for n in neighs:
            if n in curr:
                answer |= 1      # 3 ring system
            if n in next:
                num4 += 1        # 4 ring system
            if not n in found:
                next.add(n)

    answer |= min(num4,3) << 1
    found |= next

    #
    # Second level
    #
    curr = next
    next = set()

    num5 = 0
    num6 = 0
    for a in curr:
        atom = mol.GetAtomWithIdx(a)
        neighs = [n.GetIdx() for n in atom.GetNeighbors()]
        for n in neighs:
            if n in curr:
                num5 += 1        # 5 ring system, but double counted
            if n in next:
                num6 += 1        # 6 ring system
            if not n in found:
                next.add(n)

    if num5&1:
        raise Exception("Even number of 5 ring systems expected")

    answer |= min(num5>>1,3) << 3
    answer |= min(num6,3) << 5

    return answer


class Topology:
    """Establish force field components of given molecule

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule to analyze
    fixed_isomerism : bool, default False
        Flag all potential isomerisms using a default value of 1

    Notes
    -----
    Molecular force field components
    ``
                      2            3               3
                     /            /               /
        0--1     0--1         1--2            1--0
                             /                    \
                            0                      2
    
        bond    angle    proper torsion    improper torsion
    ``

    Chirality code:
       0 = Center atom is not a chiral center
      -1 = Order consistent with "clockwise" chirality
      +1 = Order consistent with "counter-clockwise chirality

    cis/trans code:
       0 = No value
      -1 = First pair of atoms in a trans configuration (opposite sides)
      +1 = First pair of atoms in a cis configuration (same side)

    Chirality "S/R" and cis/trans "E/Z" labels depend on a heristic algorithm
    for defining a specific atom order and will not necessarily match the 
    above, which uses the arbitrary atom ordering provided in the molecule.

    The option "fixed_isomerism" flags all tetra chirality and cis/trans
    bond isomerisms, even when the molecule includes no explicit annotation.
    This allows the state of the molecule to be established later based on
    atom coordinates.
    """
    def __init__(self, mol, fixed_isomerism=False):
        #
        # Ask RDkit to analyze the molecule graph and provide us all chiral
        # centers and bonds. We may want to replace this with our own algorithm 
        # at some point.
        #
        descriptor_dict = {
            Chem.rdchem.StereoDescriptor.Tet_CW: -1,
            Chem.rdchem.StereoDescriptor.Tet_CCW: +1,
            Chem.rdchem.StereoDescriptor.Bond_Trans: -1,
            Chem.rdchem.StereoDescriptor.Bond_Cis: +1,
            Chem.rdchem.StereoDescriptor.NoValue: 0
        }

        tetras = {}
        cistrans = {}
        for stereo in Chem.FindPotentialStereo(mol, cleanIt=False):
            if stereo.type == Chem.rdchem.StereoType.Atom_Tetrahedral:
                qualifier = 1 if fixed_isomerism else descriptor_dict[stereo.descriptor]
                if qualifier != 0:
                    tetras[stereo.centeredOn] = qualifier
            elif stereo.type == Chem.rdchem.StereoType.Bond_Double: 
                qualifier = 1 if fixed_isomerism else descriptor_dict[stereo.descriptor]
                if qualifier != 0:
                    bond = mol.GetBondWithIdx(stereo.centeredOn)
                    a1,a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    cistrans[(a1,a2)] = qualifier
                    cistrans[(a2,a1)] = qualifier

        #
        # Pairs of atoms that aren't "pairs"
        #
        one_four = set()

        #
        # Bonds and proper torsions
        #
        self._bonds = []
        self._propers = []
        for bond in mol.GetBonds():
            a,b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            n1 = set([n.GetIdx() for n in bond.GetBeginAtom().GetNeighbors()]) - {a,b}
            n2 = set([n.GetIdx() for n in bond.GetEndAtom().GetNeighbors()]) - {a,b}
            if b < a: 
                a,b = b,a
                n1,n2 = n2,n1
            self._bonds.append((a,b))
            one_four.add((a,b))

            code = cistrans.get((a,b),0)
            for ic,c in enumerate(sorted(n1)):
                for id,d in enumerate(sorted(n2)):
                    if c==d: continue   # don't include loop
                    #
                    # proper torsion cis/trans chirality code = 
                    #    +1 for the first neighbors of atoms a and b.
                    #    -1 for the (2nd neighbor of a) + (1st neighbor of b) if it exists
                    #    -1 for the (1st neighbor of a) + (2nd neighbor of b) if it exists
                    #    +1 for the (2nd neighbor of a) + (2nd neighbor of b) if it exists
                    # 3rd or greater neighbors should not exist for valid cis/trans bonds
                    #
                    #     1st          1st              1st         2nd
                    #        \        /                    \       /
                    #         a ==== b          or          a === b
                    #        /        \                    /       \
                    #     2nd          2nd              2nd         1st
                    #
                    self._propers.append((c,a,b,d,code * pow(-1,ic+id)))
                    one_four.add((c,d) if c < d else (d,c))

        #
        # Angles and improper torsions
        #
        self._angles = []
        self._impropers = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = sorted([a.GetIdx() for a in atom.GetNeighbors()])
            for a,b in combinations(neighbors,2):
                ring_code = build_ring_code(mol, a, idx, b)
                self._angles.append((a,idx,b,ring_code))
                one_four.add((a,b))

            tetra_code = tetras.get(idx,0)
            for a,b,c in combinations(neighbors,3):
                self._impropers.append((idx,a,b,c,tetra_code))
                tetra_code *= -1

        #
        # "pairs"
        #
        all = set(combinations(range(0,mol.GetNumAtoms()),2))
        self._pairs = list(all - one_four)

    def bonds(self):
        """Return the list of unique bonds"""
        return np.array(self._bonds) if len(self._bonds) > 0 else np.zeros((0,2))
    
    def angles(self):
        """Return the list of unique angles, with ring code appended"""
        return np.array(self._angles) if len(self._angles) > 0 else np.zeros((0,3))
    
    def pairs(self):
        """Return all nonbonded pairs
        
        Notes
        -----
        Nonbonded bonds include all unique combinations of two atoms not involved 
        in a bond, angle, or torsion
        """
        return np.array(self._pairs) if len(self._pairs) > 0 else np.zeros((0,2))

    def proper_torsions(self):
        """Return the list of unique proper torsions, with cis/trans code appended"""
        return np.array(self._propers) if len(self._propers) > 0 else np.zeros((0,5))

    def improper_torsions(self):
        """Return the list of unique improper torsions, with chirality code appended"""
        return np.array(self._impropers) if len(self._impropers) > 0 else np.zeros((0,5))

    def tetras(self):
        """Return the set of unique improper torsions associated with tetra chirality"""
        answer = [i for i in self._impropers if i[4] != 0]
        return np.array(answer) if len(answer) > 0 else np.zeros((0,5))

    def cistrans(self):
        """Return the of set of unique proper torsions associated with cis/trans isomerism"""
        answer = [i for i in self._propers if i[4] != 0]
        return np.array(answer) if len(answer) > 0 else np.zeros((0,5))

    def reorder(self, codes):
        """Reorder elements by given encoding
        
        Parameters
        ----------
        codes : list[comparable]
            The codes used to order atoms, provided in atom order
        """

        self._bonds = [
            (a,b) if codes[a] < codes[b] else (b,a) for a,b in self._bonds
        ]

        self._pairs = [
            (a,b) if codes[a] < codes[b] else (b,a) for a,b in self._pairs
        ]

        self._angles = [
            (a,b,c,code) if codes[a] < codes[c] else (c,b,a,code) for a,b,c,code in self._angles
        ]

        self._propers = [
            (d,c,b,a,e) if codes[a] > codes[d] or (codes[a] == codes[d] and codes[b] > codes[c]) else (a,b,c,d,e) for a,b,c,d,e in self._propers
        ]

        def improper_order(a,b,c,d,sign):
            if codes[b] > codes[c]:
                b,c = c,b
                sign = -sign
            if codes[b] > codes[d]:
                b,d = d,b
                sign = -sign
            if codes[c] > codes[d]:
                c,d = d,c
                sign = -sign

            return (a, b, c, d, sign)

        self._impropers = [
            improper_order(*im) for im in self._impropers
        ]
                        

def example(mol, encoding=AtomEnumeration()):
    """Return dictionary of properties for given molecule
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule of interest
    encoding : AtomEnumeration, default AtomEnumeration()
        Atom encoding instructions

    Returns
    -------
    Dict[str,varies]
        The dictionary of properties
    """
    #
    # Atom base encoding
    #
    atoms = [-1]*mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        atoms[atom.GetIdx()] = encoding.atom_encoding(atom)

    if None in atoms:
        raise Exception("Atom encoding failure: unknown atom type")

    #
    # Topology
    #
    topol = Topology(mol)
    topol.reorder(atoms)

    return {
        'atoms': atoms,
        'bonds': topol.bonds(),
        'angles': topol.angles(),
        'propers': topol.proper_torsions(),
        'impropers': topol.improper_torsions(),
        'pairs': topol.pairs(),
        'tetras': topol.tetras(),
        'cistrans': topol.cistrans()
    }


class Storage:
    """Dataset storage
    
    Parameters
    ----------
    path : string
        Path of associated sqlite3 database file
    readOnly : bool, default False
        If true, the database is opened in readonly mode
    """
    def __init__(self, path, readOnly=False):
        self.encoding = AtomEnumeration()

        self.path = path
        if readOnly:
            self.conn = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
        else:
            self.conn = sqlite3.connect(path)
            cursor = self.conn.cursor()

            cursor.execute((
                "CREATE TABLE IF NOT EXISTS molecule ("
                "id INTEGER PRIMARY KEY,"
                "name TEXT,"
                "smiles TEXT,"
                "num_atoms INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS conformer ("
                "id INTEGER PRIMARY KEY,"
                "molecule_id INTEGER REFERENCES molecule(id),"
                "sequence INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS conformer_set ("
                "label TEXT,"
                "conformer_id INTEGER REFERENCES conformer(id)"
                ")"
            ))
            #
            # The following isn't normalized, since encoding is constant over
            # all conformers of the same molecule. We just won't go to the 
            # trouble of creating a separate table just to store the encoding.
            #
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS atom ("
                "conformer_id INTEGER REFERENCES conformer(id),"
                "atom_id INTEGER,"
                "encoding INTEGER,"
                "x FLOAT,"
                "y FLOAT,"
                "z FLOAT"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS bond ("
                "molecule_id INTEGER REFERENCES molecule(id),"
                "atom1 INTEGER,"
                "atom2 INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS pair ("
                "molecule_id INTEGER REFERENCES molecule(id),"
                "atom1 INTEGER,"
                "atom2 INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS angle ("
                "molecule_id INTEGER REFERENCES molecule(id),"
                "atom1 INTEGER,"
                "atom2 INTEGER,"
                "atom3 INTEGER,"
                "ring_code INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE TABLE IF NOT EXISTS torsion ("
                "molecule_id INTEGER REFERENCES molecule(id),"
                "atom1 INTEGER,"
                "atom2 INTEGER,"
                "atom3 INTEGER,"
                "atom4 INTEGER,"
                "topology INTEGER, "         # 0 = proper, 1 = improper
                "chirality INTEGER"
                ")"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS molecule_name ON molecule(name)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS conformer_molecule_id ON conformer(molecule_id)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS conformer_set_label ON conformer_set(label)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS atom_conformer_id ON atom(conformer_id)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS bond_molecule_id ON bond(molecule_id)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS pair_molecule_id ON pair(molecule_id)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS angle_molecule_id ON angle(molecule_id)"
            ))
            cursor.execute((
                "CREATE INDEX IF NOT EXISTS torsion_molecule_id ON torsion(molecule_id)"
            ))

        self.cached_molecule = None
        self.cached_data = None

    def __getstate__(self):
        """Custom getstate, since sqlite3 doesn't like to pickle"""
        return {'path': self.path}

    def __setstate__(self, d):
        """Custom setstate, since sqlite3 doesn't like to pickle"""
        self.path = d['path']
        self.conn = sqlite3.connect(self.path)
        self.cached_molecule = None
        self.cached_data = None

    def number_tokens(self):
        """Return total number of unique atom type tokens"""
        return self.encoding.size()

    def molecule_name(self, mid):
        """Return name of molecule associated with given database id"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM molecule WHERE id = ?", (mid,))
        return cursor.fetchone()[0]

    def conformers_by_name(self, name):
        """Return all conformers associated with given molecule name"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT c.id FROM molecule m, conformer c WHERE m.id = c.molecule_id AND m.name = ?", 
            (name,)
        )
        return [r[0] for r in cursor.fetchall()]

    def conformer_sequence(self, cid):
        """Return the sequence associated with the given conformer
        
        Notes
        -----
        The sequence is a sequential number counting from zero assigned
        to the conformers of the same molecule.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT sequence FROM conformer WHERE id = ?", (cid,))
        return cursor.fetchone()[0]

    def conformers(self, label):
        """Return list of conformer ids
        
        Parameters
        ----------
        label : str
            The name of the desired conformer set

        Returns
        -------
        list[int]
            The corresponding list of conformer ids

        Notes
        -----
        The special label "ALL" returns all conformers in the data set.
        """
        if label == 'ALL':
            return pd.read_sql(
                "SELECT id FROM conformer ORDER BY id", 
                self.conn,
            )['id'].to_list()

        return pd.read_sql(
            "SELECT conformer_id FROM conformer_set WHERE label = ? ORDER BY conformer_id", 
            self.conn,
            params = (label,)
        )['conformer_id'].to_list()

    def number_atoms(self, conformer_id):
        """Return number of atoms in given conformer
        
        Parameters
        ----------
        conformer_id : int
            The conformer of interest

        Returns
        -------
        int
            The number of atoms
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT m.num_atoms FROM molecule m, conformer c WHERE m.id = c.molecule_id AND c.id = ?",(conformer_id,))
        return cursor.fetchone()[0]
    
    def rdkit_molecule(self, conformer_id, add_coords=False):
        """Return RDKit molecule for given conformer
        
        Parameters
        ----------
        conformer_id : int
            The database id of the requested conformer
        add_coords : bool, default False
            If true, 3D coordinates are added

        Returns
        -------
        A RDKit molecule, populated with explicit hydrogens. Note that the atom order
        will be scrambled.
        """
        #
        # Create output molecule
        #
        cursor = self.conn.cursor()
        cursor.execute("SELECT m.id, m.name, m.smiles FROM molecule m, conformer c WHERE m.id = c.molecule_id AND c.id = ?",(conformer_id,))
        (molecule_id, name, smiles) = cursor.fetchone()

        answer = Chem.MolFromSmiles(smiles, sanitize=False)
        answer.SetProp("_Name", name)
        answer.SetProp("mid", str(molecule_id))
        answer.SetProp("cid", str(conformer_id))

        if add_coords:
            from rdkit.Geometry import Point3D
            import networkx as nx
            from networkx.algorithms import isomorphism

            #
            # RDKit will have scrambled atom orderings during SMILES conversion.
            # Use graph matching to recover our original ordering.
            #
            answer_graph = nx.Graph()
            for atom in answer.GetAtoms():
                answer_graph.add_node(atom.GetIdx(), anum=atom.GetAtomicNum(), q=atom.GetFormalCharge())
            for bond in answer.GetBonds():
                answer_graph.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())

            #
            # Fetch database graph and coordinates
            #
            coord_graph = nx.Graph()
            coords = []
            cursor.execute("SELECT atom_id, encoding, x, y, z FROM atom WHERE conformer_id = ? ORDER BY atom_id",(conformer_id,))
            for idx, encoding, x, y, z in cursor.fetchall():
                anum, q, _ = self.encoding.properties(encoding)
                coord_graph.add_node(idx, anum=anum, q=q)
                coords.append(Point3D(x,y,z))

            cursor.execute("SELECT atom1, atom2 FROM bond WHERE molecule_id = ?",(molecule_id,))
            for a1,a2 in cursor.fetchall():
                coord_graph.add_edge(a1,a2)

            #
            # Match and assign coordinates
            #
            matcher = isomorphism.GraphMatcher(answer_graph, coord_graph, node_match=lambda x,y: x['anum'] == y['anum'] and x['q'] == y['q'])
            try:
                match = next(matcher.isomorphisms_iter())
            except StopIteration:
                raise Exception(f"SMILES does not batch database graph structure, conformer={conformer_id}, smiles={smiles}")

            conformer = Chem.Conformer(answer.GetNumAtoms())
            for i in range(answer.GetNumAtoms()):
                conformer.SetAtomPosition(i,coords[match[i]])
            answer.AddConformer(conformer)

        #
        # The following is needed to set internal properties such as ring information
        #
        Chem.SanitizeMol(answer)
        return answer

    def conformer(self, conformer_id):
        """Return graph data on given conformer
        
        Parameters
        ----------
        conformer_id : int
            The desired conformer

        Returns
        -------
        Dict[]
            A dictionary of standard graph parameters
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT molecule_id FROM conformer WHERE id = ?",(conformer_id,))
        molecule_id = cursor.fetchone()[0]

        atoms = []
        coords = []
        cursor.execute("SELECT encoding, x, y, z FROM atom WHERE conformer_id = ? ORDER BY atom_id",(conformer_id,))
        for row in cursor.fetchall():
            [encoding,x,y,z] = row
            atoms.append(encoding)
            coords.append([x,y,z])
        
        coords = np.array(coords)

        if self.cached_molecule == molecule_id:
            bonds,angles,propers,impropers,pairs = self.cached_data
        else:
            bonds = []
            cursor.execute("SELECT atom1, atom2 FROM bond WHERE molecule_id = ?",(molecule_id,))
            for row in cursor.fetchall():
                bonds.append(list(row))
            pairs = []
            cursor.execute("SELECT atom1, atom2 FROM pair WHERE molecule_id = ?",(molecule_id,))
            for row in cursor.fetchall():
                pairs.append(list(row))
            angles = []
            cursor.execute("SELECT atom1, atom2, atom3, ring_code FROM angle WHERE molecule_id = ?",(molecule_id,))
            for row in cursor.fetchall():
                angles.append(list(row))
            propers = []
            impropers = []
            cursor.execute("SELECT atom1, atom2, atom3, atom4, chirality, topology FROM torsion WHERE molecule_id = ?",(molecule_id,))
            for row in cursor.fetchall():
                if row[-1] == 0:
                    propers.append(row[:-1])
                else:
                    impropers.append(row[:-1])

            self.cached_molecule = molecule_id
            self.cached_data = (bonds,angles,propers,impropers,pairs)

        #
        # Build sublist of impropers associated with a chiral center,
        # and assign chirality according to coordinates of this conformer
        #
        tetras = []
        for imp in self.cached_data[3]:
            if imp[4] == 0: continue

            v0 = coords[imp[1],] - coords[imp[0],]
            v1 = coords[imp[2],] - coords[imp[0],]
            v2 = coords[imp[3],] - coords[imp[0],]

            if np.dot(v0,np.cross(v1,v2)) + np.dot(v2,np.cross(v0,v1)) + np.dot(v1,np.cross(v2,v0)) > 0:
                tetras.append((imp[0],imp[1],imp[2],imp[3],+1))
            else:
                tetras.append((imp[0],imp[1],imp[2],imp[3],-1))

        #
        # Build sublist of propers associated with a cis/trans isomer,
        # and assign according to coordinates of this conformer
        #
        cistrans = []
        for pro in self.cached_data[2]:
            if pro[4] == 0: continue

            u1 = coords[pro[1],] - coords[pro[0],]
            u2 = coords[pro[2],] - coords[pro[1],]
            u3 = coords[pro[3],] - coords[pro[2],]

            u1xu2 = np.cross(u1, u2)
            u2xu3 = np.cross(u2, u3)

            cistrans.append((pro[0],pro[1],pro[2],pro[3],+1 if np.dot(u1xu2,u2xu3) > 0 else -1))

        return {
            'cid': conformer_id,
            'mid': molecule_id,
            'atoms':     np.array(atoms, dtype=int),
            'coords':    np.array(coords, dtype=np.float32),
            'bonds':     np.array(bonds, dtype=int),
            'angles':    np.array(angles, dtype=int),
            'propers':   np.array(propers)   if len(propers) > 0   else np.zeros((0,5)),
            'impropers': np.array(impropers) if len(impropers) > 0 else np.zeros((0,5)),
            'pairs':     np.array(pairs)     if len(pairs) > 0     else np.zeros((0,2)),
            'tetras':    np.array(tetras)    if len(tetras) > 0    else np.zeros((0,5)),
            'cistrans':  np.array(cistrans)  if len(cistrans) > 0  else np.zeros((0,5))
        }


    def atom_encoding(self, atom):
        return self.encoding.atom_encoding(atom)

    def add_mol(self, mol, name, conformer):
        """Add rdkit molecule
        
        Parameters
        ----------
        mol : rdkit.Chem.MOL
            The molecule to add
        name : str
            The name to give this molecule
        conformer : int
            The conformer index associated with the coordinates of this molecule

        Returns
        -------
        Boolean
            True if molecule addition was successful.

        Notes
        -----
        The first conformer received for a molecule is used for topology. No check 
        is made if subsequence conformers mismatch.
        """

        #
        # Validate encoding
        #
        for atom in mol.GetAtoms():
            if self.atom_encoding(atom) is None:
                return False

        #
        # Get or create molecule
        #
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM molecule WHERE name = ?",(name,))
        row = cursor.fetchone()
        new_molecule = row is None
        if new_molecule:
            cursor.execute(
                "INSERT INTO molecule (name,smiles,num_atoms) VALUES (?,?,?)",
                (name, Chem.MolToSmiles(mol), mol.GetNumAtoms())
            )
            mol_id = cursor.lastrowid
        else:
            mol_id = row[0]

        #
        # Insert conformer
        #
        cursor.execute("INSERT INTO conformer (molecule_id,sequence) VALUES (?,?)", (mol_id,conformer))
        conformer_id = cursor.lastrowid

        #
        # Atoms
        #
        codes = {}
        for atom in mol.GetAtoms():
            coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            encoding = self.atom_encoding(atom)
            codes[atom.GetIdx()] = encoding
            cursor.execute(
                "INSERT INTO atom (conformer_id, atom_id, encoding, x, y, z) VALUES (?,?,?,?,?,?)",
                (conformer_id, atom.GetIdx(), encoding, coords[0], coords[1], coords[2])
            )

        if new_molecule:
            #
            # Save topology
            #
            # Save all elements with atom indices sorted by encoding.
            # Note that we mark all isomerisms (chirality, cis/trans) with the value of 1.
            # This will be modified once the conformer is generated based on 3D coordinates.
            #
            topol = Topology(mol, fixed_isomerism=True)
            topol.reorder(codes)

            for a,b in topol._bonds:
                cursor.execute(
                    "INSERT INTO bond (molecule_id, atom1, atom2) VALUES (?,?,?)",
                    (mol_id, a, b)
                )

            for a,b in topol._pairs:
                cursor.execute(
                    "INSERT INTO pair (molecule_id, atom1, atom2) VALUES (?,?,?)",
                    (mol_id, a, b)
                )

            for a,b,c,code in topol._angles:
                cursor.execute(
                    "INSERT INTO angle (molecule_id, atom1, atom2, atom3, ring_code) VALUES (?,?,?,?,?)",
                    (mol_id, a, b, c, code)
                )

            for a,b,c,d,e in topol._propers:
                cursor.execute(
                    "INSERT INTO torsion (molecule_id, atom1, atom2, atom3, atom4, topology, chirality) VALUES (?,?,?,?,?,0,?)",
                    (mol_id, a, b, c, d, e)
                )

            for a,b,c,d,e in topol._impropers:
                cursor.execute(
                    "INSERT INTO torsion (molecule_id, atom1, atom2, atom3, atom4, topology, chirality) VALUES (?,?,?,?,?,1,?)",
                    (mol_id, a, b, c, d, e)
                )

        return True


    def fixed_set( self, label, total_number ):
        """Declare a fixed set of given size, randomly selected by molecule
        
        Parameters
        ----------
        label : str
            The name of the set
        total_number : int
            The desired size of the set
        """
        #
        # Start by fetching all molecules by id
        #
        dt = pd.read_sql("SELECT id FROM molecule ORDER BY id", self.conn)

        #
        # Randomly reorder
        #
        ids = dt.sample(frac=1).reset_index(drop=True)['id']

        #
        # Make set
        #
        self.insert_set( label, ids[:total_number] )

    def by_name_set( self, label, selector ):
        """Declare a set selected by callable
        
        Parameters
        ----------
        label : str
            The name of the set
        selector : callable
            Pandas callable function
        """
        #
        # Start by fetching all molecules by id
        #
        dt = pd.read_sql("SELECT id, name FROM molecule ORDER BY id", self.conn)

        #
        # Filter
        #
        ids = dt.loc[dt['name'].apply(selector)]['id']

        #
        # Make set
        #
        self.insert_set( label, ids )

    def split_set( self, labels, fractions ):
        """Split our dataset into random, disparate pieces, by molecule
        
        Parameters
        ----------
        labels : List[str]
            The list of names for the sets
        fractions : List[float]
            The fraction to be assigned to the sets

        Notes
        -----
        There are no explicit checks that the fractions sum to less than
        or equal to 1.
        """

        #
        # Start by fetching all molecules by id
        #
        dt = pd.read_sql("SELECT id FROM molecule ORDER BY id", self.conn)

        #
        # Randomly reorder
        #
        ids = dt.sample(frac=1).reset_index(drop=True)['id']

        start = 0
        for label,frac in zip(labels,fractions):
            #
            # Select ids
            #
            end = int(start + frac*len(ids))
            ids_here = ids[start:end]
            start = end

            #
            # Make set
            #
            self.insert_set( label, ids_here )

    def insert_set(self, label, molecule_ids):
        """Associated the given set of molecules to a named set
        
        Parameters
        ----------
        label : str
            The name of the set
        molecule_ids : List[int]
            The molecules to add
        """
        with self.conn:
            cursor = self.conn.cursor()
            #
            # The easiest and faster method of performing this operation (AFAIK)
            # is to make a temp table, bulk loaded using executemany.
            #
            cursor.execute("CREATE TABLE temp (id)")
            cursor.executemany("INSERT INTO temp (id) VALUES (?)",[(i,) for i in molecule_ids])
            cursor.execute((
                "INSERT INTO conformer_set (label,conformer_id) "
                "SELECT ?, c.id FROM conformer c, molecule m "
                "WHERE c.molecule_id = m.id AND m.id IN (SELECT * FROM temp)"
            ), (label,))
            cursor.execute("DROP TABLE temp")

    def keep_only(self, molecule_ids):
        """Remove all but given set of molecules
        
        Parameters
        ----------
        molecule_ids : List[int]
            The molecules to keep
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("CREATE TABLE temp (id)")
            cursor.executemany("INSERT INTO temp (id) VALUES (?)",[(i,) for i in molecule_ids])

            cursor.execute("DELETE FROM bond WHERE molecule_id NOT IN (SELECT * FROM temp)")
            cursor.execute("DELETE FROM angle WHERE molecule_id NOT IN (SELECT * FROM temp)")
            cursor.execute("DELETE FROM atom WHERE conformer_id IN (SELECT id FROM conformer WHERE molecule_id NOT IN (SELECT * FROM temp))")
            cursor.execute("DELETE FROM conformer WHERE molecule_id NOT IN (SELECT * FROM temp)")
            cursor.execute("DELETE FROM molecule WHERE id NOT IN (SELECT * FROM temp)")
            cursor.execute("DROP TABLE temp")
        
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("VACUUM")


    def commit(self):
        self.conn.commit()
