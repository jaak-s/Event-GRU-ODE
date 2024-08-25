import event_gruode as eg
import pickle
import numpy as np
import pandas as pd
from types import SimpleNamespace

def test_sparse_obs():
    u = pickle.load(open("../Data/toy/toy_10k.pickle", "rb"))
    conf = SimpleNamespace(dt=0.1, add_events_to_x=False, noevent_pad=1.0, event_periods=np.array([1.0, 2.0, 3.0]))
    label_gen = eg.LabelGen(conf)

    data = eg.SparseEventData(obs=u["obs"], event=u["event"], time=u["time"], patient_id=u["patient_id"], cov=u["cov"], label_gen=label_gen, add_events_to_x=False)

    data2 = pd.read_csv("../Data/toy/toy_10k.csv")
    covs  = pd.read_csv("../Data/toy/toy_10k_covs.csv")
    covs.set_index("patient_id", inplace=True)
    edata = eg.EventData(data=data2, covs=covs, label_gen=label_gen, add_events_to_x=False)

    for i in [0, 500, 5000, 9999]:
        x = data[i]
        x2 = edata[i]

        assert (x["time"] == x2["time"]).all()
        assert (x["event"] == x2["event"]).all()
        assert np.allclose(x["x"].todense(), x2["x"])
        assert (x["label"] == x2["label"]).all()

        assert np.allclose(x["covs"], x2["covs"])

def test_sparse_obs_add_to_x():
    u = pickle.load(open("../Data/toy/toy_10k.pickle", "rb"))
    conf = SimpleNamespace(dt=0.1, add_events_to_x=False, noevent_pad=1.0, event_periods=np.array([1.0, 2.0, 3.0]))
    label_gen = eg.LabelGen(conf)

    data = eg.SparseEventData(obs=u["obs"], event=u["event"], time=u["time"], patient_id=u["patient_id"], cov=u["cov"], label_gen=label_gen, add_events_to_x=True)

    data2 = pd.read_csv("../Data/toy/toy_10k.csv")
    covs  = pd.read_csv("../Data/toy/toy_10k_covs.csv")
    covs.set_index("patient_id", inplace=True)
    edata = eg.EventData(data=data2, covs=covs, label_gen=label_gen, add_events_to_x=True)

    for i in [0, 500, 5000, 9999]:
        x = data[i]
        x2 = edata[i]

        assert (x["time"] == x2["time"]).all()
        assert (x["event"] == x2["event"]).all()
        assert np.allclose(x["x"].todense(), x2["x"])
        assert (x["label"] == x2["label"]).all()

        assert np.allclose(x["covs"], x2["covs"])

