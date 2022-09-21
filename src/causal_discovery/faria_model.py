from causal_discovery.faria.utils.experiment_utils import run_discover
from torch.utils.data import DataLoader
import networkx as nx

hyperparameter_defaults = dict(
    lr=1e-2,
    batch_size=8000,
    epochs=500,
    ro_percentage=0.00001,
    mollify_percentage=0.00001,
    mixer_percentage=1.0,
    temperature_percentage=1.5,
    alpha=0.1,
    hdim=264,
    k=264,
    prior=-1,# 44 # 46 #4706
    graph_prior=0.5,#0.496
    beta1=0.9,
    beta2=0.99,
    wd=1e-6,
    lh=0.5,
    mlpdim=64,
    mlplayers=2,
    use_z_entropy=False,
    modeltype="linear",  # can also be nonlinear or flow
    method="constraint",
    estimator="b-gst",
    weight_sparsity=0.0,
    target_sparsity=0.0,
    known=False,
    perfect=True, # TODO: as init param?
    atomic=False,
    hard=True,
    oracle=False,
    ssl=False
)


class Faria:
    def __init__(self,
                 modeltype,
                 hparams=hyperparameter_defaults,
                 device='cpu',
                 seed=-1,
                 batch_size=128):

        hparams['model_seed'] = seed
        hparams['modeltype'] = modeltype
        self.hparams = hparams
        self.device = device
        self.batch_size = batch_size

    def predict(self, cd_input: tuple):

        variables, data = cd_input
        self.hparams['n'] = len(variables)
        self.hparams['i'] = len(variables) + 1 # this assumes the number of intervention targets as known

        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        G = run_discover(makeup=self.hparams,
                         data=dataloader,
                         device=self.device,
                         debug=False,
                         use_wandb=False)

        #print(G)
        G = G.detach().cpu().numpy()

        pred_graph = nx.from_numpy_array(G, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)
