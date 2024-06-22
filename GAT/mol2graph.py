# -*- coding: utf-8 -*-
# @Time : 2023/7/11 23:32
# @Author : Crush
# @Version: 3.9.5
import torch
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from typing import Union, List, Tuple
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data

try:
    with open("./atomSymbol_encoder.pickle", "rb") as f1:
        atomsymbol_encoder = pickle.load(f1)
    with open("./atomDegree_encoder.pickle", "rb") as f2:
        atomDegree_encoder = pickle.load(f2)
    with open("./hybridization_encoder.pickle", "rb") as f3:
        atomHybridization_encoder = pickle.load(f3)
    with open("./hydrogens_encoder.pickle", "rb") as f4:
        atomHydrogens_encoder = pickle.load(f4)
    with open("./chiralType_encoder.pickle", "rb") as f5:
        chiralType_encoder = pickle.load(f5)
    with open("./bondType_encoder.pickle", "rb") as f6:
        bondType_encoder = pickle.load(f6)
    with open("./bondStereo_encoder.pickle", "rb") as f7:
        bondStereo_encoder = pickle.load(f7)

except Exception:
    print('特征编码器未能正常加载!!')


class Lsh_MolDataset(InMemoryDataset):

    def __init__(self, root='./lsd1-gnn', transform=None, pre_transform=None):
        super(Lsh_MolDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['lsd1-graph-mol.pt']

    def download(self):
        pass

    def process(self):
        mol_data_list = []
        # load molecules
        df = pd.read_excel('../train_test_molecules.csv', sheet_name='all_molecules')
        mols = [Chem.MolFromSmiles(x) for x in df.SMILE]
        labels = df.Active
        n = 0 
        for mol in tqdm(mols):
            # n number for label
            atoms = mol.GetAtoms()
            bonds = mol.GetBonds()
            node_features = []
            edge_features = []
            begin_atom = []
            end_atom = []

            for atom in atoms:
                symbol_features = atomsymbol_encoder.transform([[atom.GetSymbol()]]).squeeze()
                degree_features = atomDegree_encoder.transform([[atom.GetTotalDegree()]]).squeeze()
                formal_charge = atom.GetFormalCharge()
                radical_electrons = atom.GetNumRadicalElectrons()
                hybridization = str(atom.GetHybridization())
                hybridization_features = atomHybridization_encoder.transform([[hybridization]]).squeeze()
                aromaticity = int(atom.GetIsAromatic())
                hydrogens_features = atomHydrogens_encoder.transform([[atom.GetTotalNumHs()]]).squeeze()
                chiral_type = str(atom.GetChiralTag())
                chiral_features = chiralType_encoder.transform([[chiral_type]]).squeeze()
                isinring = int(atom.IsInRing())
                atom_features = np.concatenate([symbol_features, degree_features, [formal_charge],
                                                [radical_electrons], hybridization_features, [aromaticity],
                                                hydrogens_features, chiral_features, [isinring]])
                node_features.append(atom_features)

            for bond in bonds:

                begin_atom.append(bond.GetBeginAtomIdx())
                end_atom.append(bond.GetEndAtomIdx())
                bondType = str(bond.GetBondType())
                bondType_features = bondType_encoder.transform([[bondType]]).squeeze()
                bond_conjugate = int(bond.GetIsConjugated())
                bond_inring = int(bond.IsInRing())
                bondStereo = str(bond.GetStereo())
                bondStereo_features = bondStereo_encoder.transform([[bondStereo]]).squeeze()
                bond_features = np.concatenate(
                    [bondType_features, [bond_conjugate], [bond_inring], bondStereo_features])
                edge_features.append(bond_features)

            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor([begin_atom, end_atom], dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            label = labels[n]
            # y = torch.FloatTensor([label])
            y = torch.tensor([label], dtype=torch.float)
            n += 1
            mol_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=y)
            mol_data = T.ToUndirected()(mol_data)
            mol_data_list.append(mol_data)

        data, slices = self.collate(mol_data_list)
        torch.save((data, slices), self.processed_paths[0])





