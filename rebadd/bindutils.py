import os
import sys
import requests
import torch
import pickle
from rdkit.Chem import MolFromSmiles, MolToSmiles
from torch.utils.data import DataLoader

BA_MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
BA_MODULE_MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "BA_module"))
sys.path = sys.path if BA_MODULE_PATH in sys.path else [BA_MODULE_PATH] + sys.path
sys.path = sys.path if BA_MODULE_MODULE_PATH in sys.path else [BA_MODULE_MODULE_PATH] + sys.path

from BA_module.module.helpers import Mycall, load_checkpoint_eval
from BA_module.module.DNN import Test


## Global variables
global BINDUTIL_REGRESSOR, BINDUTIL_MC
BINDUTIL_REGRESSOR = None
BINDUTIL_MC = None


def normalize_SMILES(smi):
    try:
        mol = MolFromSmiles(smi)
        smi_rdkit = MolToSmiles(
                        mol,
                        isomericSmiles=False,   # modified because this option allows special tokens (e.g. [125I])
                        kekuleSmiles=False,     # default
                        rootedAtAtom=-1,        # default
                        canonical=True,         # default
                        allBondsExplicit=False, # default
                        allHsExplicit=False     # default
                    )
    except:
        smi_rdkit = ''
    return smi_rdkit

    
## prepare re-training BA model
def prepareBReward(filepath_regressor=os.path.join(BA_MODULE_PATH, "./BA_module/model/train_merged.pth"),
                   filepath_protein_voca=os.path.join(BA_MODULE_PATH, "./BA_module/model/Sequence_voca.pkl"),
                   filepath_smiles_voca=os.path.join(BA_MODULE_PATH, "./BA_module/model/SMILES_voca.pkl"),
                   device=None, use_cuda=False):
    if device is None:
        device = torch.device('cpu')

    ### Loading the pretrained model
    regressor = load_checkpoint_eval(filepath_regressor, use_cuda, device)
    regressor.to(device)
    
    ### Loading caller for inputs
    with open(filepath_protein_voca, "rb") as f:
        Protein_voca = pickle.load(f)

    with open(filepath_smiles_voca, "rb") as f:
        SMILES_voca = pickle.load(f)

    mc = Mycall(Protein_voca, SMILES_voca, use_cuda)
    return regressor, mc


def check_and_build(device=None, use_cuda=None):
    global BINDUTIL_REGRESSOR, BINDUTIL_MC
    if BINDUTIL_REGRESSOR is None and BINDUTIL_MC is None:
        BINDUTIL_REGRESSOR, BINDUTIL_MC = prepareBReward(device=device, use_cuda=use_cuda)
    return BINDUTIL_REGRESSOR, BINDUTIL_MC


class BAScorer:
    def __init__(self, device=None, use_cuda=None):
        self.use_cuda = use_cuda
        self.device = device
        self.reg, self.mc = check_and_build(device=device, use_cuda=use_cuda)
        
    def __call__(self, smi):
        smi = normalize_SMILES(smi)
        
        if smi == '':
            return 0.
        else:
            loader = DataLoader(dataset=[(self.pseq, smi)],
                                batch_size=1,
                                collate_fn=self.mc,
                                pin_memory=self.use_cuda,
                                shuffle=False, drop_last=False)

            with torch.no_grad():
                batch = next(iter(loader))
                out = self.reg(*batch) # out.shape = (batch, 1)
                score = out.item()

            ## memory free
            del out, batch
            return score


class BAScorerBCL2(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCL2, self).__init__(*args, **kwargs)
        self.pseq = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK"
        self.pid = "P10415"
    

class BAScorerBCLXL(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCLXL, self).__init__(*args, **kwargs)
        self.pseq = "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK"
        self.pid = "Q07817"
        
        
class BAScorerBCLW(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCLW, self).__init__(*args, **kwargs)
        self.pseq = "MATPASAPDTRALVADFVGYKLRQKGYVCGAGPGEGPAADPLHQAMRAAGDEFETRFRRTFSDLAAQLHVTPGSAQQRFTQVSDELFQGGPNWGRLVAFFVFGAALCAESVNKEMEPLVGQVQEWMVAYLETQLADWIHSSGGWAEFTALYGDGALEEARRLREGNWASVRTVLTGAVALGALVTVGAFFASK"
        self.pid = "Q92843"


## modified in 2021.06.25
## predict BA from regressor
def calc_binding_affinity(smiles, aminoseq, regressor, mc):
    pSeq_SMILES_list = []
    pSeq_SMILES_list.append((aminoseq, smiles))
    test_loader = DataLoader(dataset=pSeq_SMILES_list, batch_size=400, collate_fn=mc)
    
    test_module = Test(regressor, test_loader)
    ba_reg = test_module.predict()[0]
    return ba_reg


class UBLBioDTA(object):
    def __init__(self, device, use_cuda):
        self.device = device
        self.use_cuda = use_cuda
        self.regressor, self.mc = self._init_BA_regressor()
        self.regressor.eval()
        
    def __call__(self, aminoseq, list_ligand_ids, list_ligand_seqs, batch_size=500):
        ## Initialize scores
        scores = dict()
        ## Input
        pSeq_SMILES_list = []
        for smi in list_ligand_seqs:
            pSeq_SMILES_list.append((aminoseq, smi))
        ## Pytorch DataLoader
        test_loader = DataLoader(dataset=pSeq_SMILES_list,
                                 batch_size=batch_size,
                                 collate_fn=self.mc,
                                 shuffle=False, pin_memory=False, drop_last=False)
        ## Pytorch Prediction
        list_outs = []
        with torch.no_grad():
            for batch in test_loader:
                out = self.regressor(*batch) # out.shape = (batch, 1)
                list_outs.append(out)
        list_ba_scores = torch.cat(list_outs, dim=0)
        list_ba_scores = list_ba_scores.view(-1).detach().cpu().numpy() # list_ba_scores.shape = (batch,)
        ## Save
        for ligand_id, ba_score in zip(list_ligand_ids, list_ba_scores):
            scores[ligand_id] = ba_score
        return scores
        
    def _init_BA_regressor(self):
        filepath_regressor    = os.path.join(BA_MODULE_PATH, "./BA_module/model/train_merged.pth")
        filepath_protein_voca = os.path.join(BA_MODULE_PATH, "./BA_module/model/Sequence_voca.pkl")
        filepath_smiles_voca  = os.path.join(BA_MODULE_PATH, "./BA_module/model/SMILES_voca.pkl")
        regressor, mc = prepareBReward(filepath_regressor, filepath_protein_voca, filepath_smiles_voca, device=self.device, use_cuda=self.use_cuda)
        return regressor, mc