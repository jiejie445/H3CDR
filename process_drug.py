#get drug features using Deepchem library
import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import hickle as hkl


# drug_smiles_file='data/228drug_pubchem_smiles.txt'
# save_dir='data/228drug_graph_feat'
# pubchemid2smile = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open(drug_smiles_file, 'r', encoding='utf-8').readlines()}
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# molecules = []
# for each in pubchemid2smile.keys():
#     print(f"Processing PubChem ID: {each}")
#     molecules=[]
#     molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
#     featurizer = dc.feat.graph_features.ConvMolFeaturizer()
#     mol_object = featurizer.featurize(molecules)
#     features = mol_object[0].atom_features
#     degree_list = mol_object[0].deg_list
#     adj_list = mol_object[0].canon_adj_list
#     hkl.dump([features,adj_list,degree_list],'%s/%s.hkl'%(save_dir,each))


drug_smiles_file='data/24drug_pubchem_smiles.txt'
save_dir='data/24drug_graph_feat'
pubchemid2smile = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open(drug_smiles_file, 'r', encoding='utf-8').readlines()}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
molecules = []
for each in pubchemid2smile.keys():
    print(f"Processing PubChem ID: {each}")
    molecules=[]
    molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize(molecules)
    features = mol_object[0].atom_features
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list
    hkl.dump([features,adj_list,degree_list],'%s/%s.hkl'%(save_dir,each))

