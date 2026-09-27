"""Load the maleCNS v1.0 connectome (Janelia FlyEM / Cambridge, 2025).

maleCNS is the complete central nervous system of an adult *male*
Drosophila melanogaster: brain, optic lobes and ventral nerve cord, about
165,000 traced neurons and 124 million synapses between them.

Public release files are fetched from ``gs://flyem-male-cns/v1.0`` over
HTTPS (no account needed). We use three flat tables:

* body annotations  (cell type, class, side ...)
* neurotransmitter predictions  (sign of each neuron's synapses)
* traced-only connection weights  (synapse counts between neurons)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "transmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "malecns-v1.0")

# Fast synaptic sign by transmitter. Glutamate is inhibitory in the fly CNS
# (GluCl), histamine too (HisCl). Monoamines act through slow G-protein
# coupled receptors and are treated as neuromodulatory (no fast effect).
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    "tyramine": 0.0,
}


def download(data_dir: str = DEFAULT_DATA_DIR, verbose: bool = True):
    os.makedirs(data_dir, exist_ok=True)
    for f in FILES.values():
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            continue
        if verbose:
            print(f"downloading {f} ...", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(BASE_URL + f, tmp)
        os.replace(tmp, path)
    return data_dir


@dataclass
class Connectome:
    body_ids: np.ndarray          # (N,) int64 maleCNS body IDs, index = neuron index
    ann: pd.DataFrame             # annotations aligned to body_ids
    nt: np.ndarray                # (N,) transmitter name used for the sign
    sign: np.ndarray              # (N,) +1 / -1 / 0
    W: sp.csc_matrix              # (N, N) signed synapse counts, W[post, pre]
    n_synapses: int

    @property
    def n(self):
        return len(self.body_ids)

    def index(self, body_ids):
        pos = np.searchsorted(self.body_ids, body_ids)
        ok = (pos < self.n) & (self.body_ids[np.minimum(pos, self.n - 1)] == body_ids)
        return pos[ok]

    def select(self, type=None, side=None, cls=None, side_col="somaSide", query=None):
        """Neuron indices matching a cell type (str / list), side and class."""
        a = self.ann
        m = np.ones(self.n, bool)
        if type is not None:
            types = [type] if isinstance(type, str) else list(type)
            m &= a["type"].isin(types).to_numpy()
        if cls is not None:
            m &= (a["class"] == cls).to_numpy()
        if side is not None:
            m &= (a[side_col] == side).to_numpy()
        if query is not None:
            m &= query
        return np.nonzero(m)[0]

    def outputs(self, idx):
        """Summed signed output weights of neurons idx onto every neuron."""
        return np.asarray(self.W[:, idx].sum(axis=1)).ravel()

    def inputs(self, idx):
        """Summed signed input weights from every neuron onto neurons idx."""
        return np.asarray(self.W[idx, :].sum(axis=0)).ravel()


VNC_SUPERCLASSES = {
    "vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_endocrine", "vnc_tbc",
    "vnc_sensory_tbc", "ascending_neuron", "sensory_ascending", "sensory_ascending_tbc",
    "efferent_ascending",
}


def load(data_dir: str = DEFAULT_DATA_DIR, min_weight: int = 2, region: str = "brain",
         exclude_kc_kc: bool = True, exclude_al_excitatory_loops: bool = True,
         verbose: bool = True) -> Connectome:
    """Load maleCNS as a signed connectivity matrix.

    region: "brain" (default) keeps the central brain, optic lobes and the
    descending neurons, whose firing is the interface to the body. The fly's
    ventral nerve cord runs six fly legs, wings and halteres; the humanoid has
    its own "spinal cord" (the learned locomotion controller), so the fly's is
    left out, together with the ascending neurons that would report the
    state of fly legs that do not exist. "cns" keeps everything.

    exclude_kc_kc: drop Kenyon cell -> Kenyon cell synapses (0.7% of synapses).
    They are mostly axo-axonic contacts in the mushroom-body lobes; a point-neuron
    model would wrongly treat them as somatic drive, which makes the ~4,000 Kenyon
    cells recruit each other into seizure-like runaway firing.

    exclude_al_excitatory_loops: drop excitatory connections among antennal-lobe
    projection and local neurons (1.1% of synapses). These are largely
    dendro-dendritic contacts inside glomeruli; as point-neuron synapses they form
    a loop that, once any stimulus kicks it, keeps ~8,000 neurons firing
    indefinitely. Inhibitory (GABA/glutamate) local-neuron connections are kept.
    """
    download(data_dir, verbose)
    p = lambda k: os.path.join(data_dir, FILES[k])
    w = pd.read_feather(p("weights"), columns=["body_pre", "body_post", "weight"])
    w = w[w["weight"] >= min_weight]
    ids = np.union1d(w["body_pre"].to_numpy(), w["body_post"].to_numpy())
    ann = pd.read_feather(p("annotations")).set_index("bodyId")
    ann = ann.reindex(ids)
    if region == "brain":
        vnc = ann["superclass"].isin(VNC_SUPERCLASSES).to_numpy() | ann["superclass"].isna().to_numpy()
        ids = ids[~vnc]
        ann = ann.loc[ids]
        w = w[np.isin(w["body_pre"].to_numpy(), ids) & np.isin(w["body_post"].to_numpy(), ids)]
    elif region != "cns":
        raise ValueError(region)
    ann.index.name = "bodyId"
    nt = pd.read_feather(p("transmitters"),
                         columns=["body", "consensus_nt", "predicted_nt", "celltype_predicted_nt"]).set_index("body")
    nt = nt.reindex(ids)
    name = nt["consensus_nt"].where(nt["consensus_nt"].isin(list(NT_SIGN)), nt["celltype_predicted_nt"])
    name = name.where(name.isin(list(NT_SIGN)), nt["predicted_nt"])
    name = name.where(name.isin(list(NT_SIGN)), "acetylcholine").to_numpy().astype(str)
    sign = np.array([NT_SIGN[x] for x in name])
    pre = np.searchsorted(ids, w["body_pre"].to_numpy())
    post = np.searchsorted(ids, w["body_post"].to_numpy())
    cls = ann["class"].to_numpy()
    keep = np.ones(len(pre), bool)
    if exclude_kc_kc:
        kc = cls == "Kenyon_Cell"
        keep &= ~(kc[pre] & kc[post])
    if exclude_al_excitatory_loops:
        al = (cls == "ALPN") | (cls == "ALLN")
        keep &= ~(al[pre] & al[post] & (sign[pre] > 0))
    pre, post, w = pre[keep], post[keep], w[keep]
    vals = w["weight"].to_numpy().astype(np.float32) * sign[pre].astype(np.float32)
    W = sp.csc_matrix((vals, (post, pre)), shape=(len(ids), len(ids)), dtype=np.float32)
    W.eliminate_zeros()
    if verbose:
        print(f"maleCNS v1.0 ({region}): {len(ids):,} neurons, {W.nnz:,} connections, "
              f"{int(w['weight'].sum()):,} synapses (edges >= {min_weight} synapses)", flush=True)
    return Connectome(ids, ann.reset_index(), name, sign, W, int(w["weight"].sum()))
