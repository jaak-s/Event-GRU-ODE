from event_gruode.models import EventGRUBayes
from event_gruode.utils import compute_metrics
import event_gruode as eg
import torch
import optuna
import numpy as np
import argparse
import pandas as pd
import os
import json
import pickle
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy.special

def df_to_path(df, add_events):
    if add_events: drop_cols = ["patient_id", "time"]
    else:          drop_cols = ["patient_id", "time", "event"]
    return {
        "time":  df.time.values,
        "event": df.event.values,
        "x":     df.drop(columns=drop_cols).values.astype(np.float32)
    }

def times_to_idx(times, dt):
    return np.round(times / dt).astype(np.int64)

class LabelGen(object):
    """Generates labels.
    Args:
        conf.dt               dt for generating true labels
        conf.add_events_to_x  whether to add event labels to x
        conf.noevent_pad      extra padding of no-events (in time), if last observation was no event
        conf.event_periods    prediction periods for events
    """
    def __init__(self, conf):
        self.dt = conf.dt
        self.add_events_to_x = conf.add_events_to_x
        self.noevent_pad     = conf.noevent_pad
        self.event_periods   = np.array(conf.event_periods)
        self.event_periods_idx = times_to_idx(self.event_periods, self.dt)
        self.label_buffer    = conf.label_buffer

    def to_labels(self, times, events, return_eval=True):
        """
        Args:
            times    event/noevent time
            events   whether there was an event or not
        Returns:
            labels
            labels_eval
            labels_eval_tx
        """
        ## TODO: add asserts, that times is increasing,
        ## and that labels is creater than times

        idx = times_to_idx(times, self.dt)
        num_periods = self.event_periods.shape[0]
        assert self.noevent_pad >= 0, "noevent_pad must be non-negative."
        if events[-1] == 0:
            ## add padding
            pad_size = times_to_idx(self.noevent_pad, self.dt)
            if pad_size == 0:
                ## adding nan as an extra label at the end
                labels = np.zeros((idx[-1] + 1, num_periods), dtype=np.float32)
                labels[-1] = np.nan
            else:
                labels = np.zeros((idx[-1] + pad_size, num_periods), dtype=np.float32)
        else:
            ## no padding (last was event)
            labels = np.zeros((idx[-1], num_periods), dtype=np.float32)

        ## adding events
        for i, e in zip(idx, events):
            if e == 0:
                continue
            start = np.clip(i - self.event_periods_idx, a_min=0, a_max=None)
            for p in range(labels.shape[1]):
                labels[start[p]:i, p] = 1

        ## marking last time points as missing (up to period len - label_buffer):
        buff_len = times_to_idx(self.label_buffer, self.dt)
        for i, period_len_idx in enumerate(self.event_periods_idx):
            if buff_len < period_len_idx:
                ## keeping buff_len of data from the beginning
                period_len_idx -= buff_len
                labels[-period_len_idx:,i] = np.nan

        if not return_eval:
            return labels

        labels_eval_tx = idx[events == 0]
        labels_eval    = labels[labels_eval_tx]
        
        return labels, labels_eval, labels_eval_tx

def to_sparse(obs_df, folds, covs=None):
    """
    Formats dense matrices to sparse matrix.
    Args:
      obs_df      matrix of observations, including cols 'time', 'patient_id', 'event'
      cov         patient covariate matrix (P x C)
      folds       2 column DataFrame with columns ['fold', 'patient_id'] (P x 2)
    Return dictionary with:
      event       array (N)
      obs         csr matrix (N x F)
      time        array (N)
      patient_id  array (N)
      cov         2d array (P x C)
      folds       two-column DataFrame (P x C)
    """
    if covs is not None:
        assert "patient_id" in covs.columns, "covs must have 'patient_id' column"
        covs = covs.set_index("patient_id")
        covs.sort_index(axis=0, inplace=True)
        covs = covs.values.astype(np.float32)
        assert cov.shape[0] == folds.shape[0], "cov is [P x C] matrix, and folds is [P x 2] "
    assert "patient_id" in folds, "folds must have column 'patient_id'"
    assert "fold" in folds, "folds must have column 'fold'"
    res = {
        "obs":        csr_matrix(obs_df.drop(columns=["patient_id", "time", "event"]).values.astype(np.float32)),
        "event":      obs_df.event.astype(np.float32).values,
        "time":       obs_df.time.values,
        "patient_id": obs_df.patient_id.values,
        "cov":        covs,
        "folds":      folds,
    }
    return res

## pickled data:
#  
#  Where N = 127122003, D = 6094 (including "event"), P = 2477647, C = 28, num_folds = 10.
class SparseEventData(torch.utils.data.Dataset):
    """
    Args:
      obs         csr matrix of observations (N x D)
      event       vector of binary event values (N) - whether event happened at the observation
      time        vector of times (N)
      patient_id  vector of patient_ids (N)
      cov         patient covariate matrix (P x C, where P is the number of patients and C is the number of covariates)
    """
    def __init__(self, obs, event, time, patient_id, cov, label_gen, idx=None, add_events_to_x=True):
        assert obs.shape[0] == event.shape[0], "rows in 'obs' must be equal to the length of 'event'"
        assert obs.shape[0] == time.shape[0], "rows in 'obs' must be equal to the length of 'time'"
        assert obs.shape[0] == patient_id.shape[0], "rows in 'obs' must be equal to the length of 'patient_id'"

        if idx is not None:
            keep  = np.isin(patient_id, idx)
            obs   = obs[keep]
            event = event[keep]
            time  = time[keep]
            patient_id = patient_id[keep]

        if (patient_id[1:] < patient_id[:-1]).any():
            raise ValueError("patient_id's are not sorted. Only sorted patient_id's allowed.")
        if (cov is not None) and (patient_id[-1] >= cov.shape[0]):
            raise ValueError(f"Patient id larger than number of rows in cov ({cov.shape[0]}).")

        self.obs        = obs
        self.obs.data   = self.obs.data.astype(np.float32)
        self.event      = event.astype(np.float32)
        self.time       = time
        self.patient_id = idx

        self.patient_segm = np.concatenate([
            [0],
            np.where(patient_id[1:] > patient_id[:-1])[0] + 1,
            [patient_id.shape[0]],
        ])
        if len(patient_id) == 0 or cov is None:
            self.conf = None
        else:
            kept     = patient_id[self.patient_segm[:-1]]
            self.cov = cov[kept].astype(np.float32)

        if idx is not None:
            assert len(kept) == len(idx), f"Following patients do not have any observations:\n{np.setdiff1d(idx, kept)}."

        if add_events_to_x:
            self.obs = scipy.sparse.hstack([self.event[:,None], self.obs], format="csr")

        ## generating labels
        self.label = []
        for i in tqdm(range(self.patient_segm.shape[0] - 1)):
            start  = self.patient_segm[i]
            end    = self.patient_segm[i + 1]
            labels = label_gen.to_labels(times=self.time[start:end], events=self.event[start:end], return_eval=False)
            self.label.append(labels)

    def __getitem__(self, idx):
        start = self.patient_segm[idx]
        end   = self.patient_segm[idx + 1]
        out = {
            "event": self.event[start:end],
            "time":  self.time[start:end],
            "x":     self.obs[start:end],
            "label": self.label[idx],
            "patient_id": self.patient_id[idx],
        }
        if self.cov is not None:
            out["covs"] = self.cov[idx]
        return out

    def __len__(self):
        return self.patient_segm.shape[0] - 1

    @property
    def cov_size(self):
        if self.cov is None: return 0
        return self.cov.shape[1]

    @property
    def input_size(self):
        return self.obs.shape[1]

class EventData(torch.utils.data.Dataset):
    """
    Args:
        data    should have have 'patient_id', 'time' and 'event' columns, rest are x
        covs    should have 'patient_id' as index
        conf    conf object (for dt, noevent_pad, add_events_to_x, event_periods)
    """
    def __init__(self, data, covs, label_gen, idx=None, add_events_to_x=True):
        if idx is not None:
            data = data[data.patient_id.isin(idx)]
        self.pids = []
        self.data = []

        self.label_gen = label_gen

        self.labels = []
        self.labels_eval = []
        self.labels_eval_tx = []

        for pid, dfi in tqdm(data.groupby("patient_id")):
            self.pids.append(pid)
            self.data.append(df_to_path(dfi, add_events_to_x))
            labels, labels_eval, labels_eval_tx = label_gen.to_labels(
                times = dfi["time"].values,
                events = dfi["event"].values,
            )
            self.labels.append(labels)
            self.labels_eval.append(labels_eval)
            self.labels_eval_tx.append(labels_eval_tx)

        self.pids = np.array(self.pids)
        if covs is not None:
            assert "patient_id" not in covs.columns, "Please set 'patient_id' as index."
            self.covs = covs.loc[self.pids].values.astype(np.float32)
        else:
            self.covs = None

    def __getitem__(self, idx):
        out = {
            "event": self.data[idx]["event"],
            "time":  self.data[idx]["time"],
            "x":     self.data[idx]["x"],
            "label": self.labels[idx],
            "label_eval": self.labels_eval[idx],
            "label_eval_tx": self.labels_eval_tx[idx],
            "patient_id": idx,
        }
        if self.covs is not None:
            out["covs"] = self.covs[idx]
        return out

    def __len__(self):
        return len(self.data)

    @property
    def cov_size(self):
        if self.covs is None: return 0
        return self.covs.shape[1]

    @property
    def input_size(self):
        return self.data[0]["x"].shape[1]

def collate_events(batch):
    """Collate several patient data together."""
    data = {}
    ## processing covariates
    if batch[0].get("covs", None) is not None:
        data["covs"] = np.row_stack([b["covs"] for b in batch])
    else:
        data["covs"] = np.zeros((len(batch), 1), dtype=np.float32)

    ## processing event data
    if type(batch[0]["x"]) == scipy.sparse.csr.csr_matrix:
        ## converting matrices to dense
        for b in batch:
            b["x"] = b["x"].todense()

    for key in ["time", "event", "x"]:
        data[key] = np.concatenate([b[key] for b in batch])

    sizes = [len(b["time"]) for b in batch]
    data["sample_ids"] = np.repeat(np.arange(len(batch)), sizes)

    ## re-ordering
    sorting_idx = np.argsort(data["time"])
    for key in ["time", "event", "x", "sample_ids"]:
        data[key] = data[key][sorting_idx]

    ## unique time points:
    data["time_ptr"] = np.concatenate([
        [0],
        np.where(data["time"][1:] != data["time"][:-1])[0] + 1,
        [len(data["time"])],
    ])
    data["time_uniq"] = data["time"][data["time_ptr"][:-1]]

    ## padding labels:
    labels = [torch.from_numpy(b["label"]) for b in batch]
    data["labels"] = torch.nn.utils.rnn.pad_sequence(labels, padding_value=np.nan)
    data["patient_ids"] = torch.LongTensor([b["patient_id"] for b in batch])

    #rows = np.repeat(np.arange(len(batch)), [len(b["label_eval_tx"]) for b in batch]) 
    #data["labels_eval_idx"] = torch.stack([
    #    torch.from_numpy(rows),
    #    torch.from_numpy(np.concatenate([b["label_eval_tx"] for b in batch])),
    #])
    #data["labels_eval_val"] = torch.from_numpy(np.concatenate([b["label_eval"] for b in batch]))
    #assert data["labels_eval_val"].shape[0] == data["labels_eval_idx"].shape[1]

    return data

def times_to_tensor(time_ids, sample_ids, shape):
    """returns 2D tensor with 1's at the places specified by times and sample_ids"""
    x = torch.zeros(shape, device=time_ids.device)
    x[time_ids, sample_ids] = 1.0
    return x

class EventRun(object):
    """
    Args:
        data       DataFrame with events
        covs       covariate data (DataFrame num_patients x cov_size), can be None
        folds      dataframe with patient ids and folds
        conf       conf object
    """

    def __init__(self, data, covs, folds, conf, trial=None):
        self.conf    = conf
        self.device  = torch.device(conf.device)
        self.trial   = trial
        dt = conf.dt

        if covs is not None:
            assert "patient_id" in covs.columns, "covs must have 'patient_id' column"
            covs = covs.set_index("patient_id")
            
        self.idx_tr  = folds.query("fold != @conf.fold_va and fold != @conf.fold_te").patient_id.values
        self.idx_va  = folds.query("fold == @conf.fold_va").patient_id.values
        self.idx_te  = folds.query("fold == @conf.fold_te").patient_id.values

        assert len(self.idx_tr) > 0, f"Training folds have no samples. Cannot train."
        if conf.fold_va >= 0:
            assert len(self.idx_va) > 0, f"Validation fold ({conf.fold_va}) has 0 samples. Cannot evaluate."

        add_e   = conf.add_events_to_x
        label_gen = LabelGen(conf)
        if type(data) == dict:
            ## sparse data
            if "event" not in data:
                data["event"] = np.array(data["obs"][:,0].todense()).flatten()
                data["obs"]   = data["obs"][:,1:]
            datax = {
                "obs":        data["obs"],
                "event":      data["event"],
                "time":       data["time"],
                "patient_id": data["patient_id"],
                "cov":        data["cov"],
            }
            data_tr = SparseEventData(**datax, add_events_to_x=add_e, idx=self.idx_tr, label_gen=label_gen)
            data_va = SparseEventData(**datax, add_events_to_x=add_e, idx=self.idx_va, label_gen=label_gen)
            data_te = SparseEventData(**datax, add_events_to_x=add_e, idx=self.idx_te, label_gen=label_gen)
        else:
            data_tr = EventData(data, covs, add_events_to_x=add_e, idx=self.idx_tr, label_gen=label_gen)
            data_va = EventData(data, covs, add_events_to_x=add_e, idx=self.idx_va, label_gen=label_gen)
            data_te = EventData(data, covs, add_events_to_x=add_e, idx=self.idx_te, label_gen=label_gen)

        ## then create loader_tr, loader_va, loader_te
        self.loader_tr = self.data_loader(data_tr, limit_batches=self.conf.limit_train_batches, shuffle=True)
        self.loader_va = self.data_loader(data_va, limit_batches=self.conf.limit_val_batches, shuffle=False)
        self.loader_te = self.data_loader(data_te, limit_batches=self.conf.limit_test_batches, shuffle=False)

        self.conf.cov_size    = data_tr.cov_size
        self.conf.input_size  = data_tr.input_size
        self.conf.output_size = len(self.conf.event_periods)
        
        model = EventGRUBayes(conf).to(self.device)
        print(f"Covariate size:   {self.conf.cov_size}")
        print(f"Input (obs) size: {self.conf.input_size}")
        print(f"Output size:      {self.conf.output_size}")
        print("[[Model]]")
        print(model)
        self.model = torch.jit.script(model)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=conf.lr_steps, gamma=conf.lr_alpha)

        self.writer = None
        if conf.save_board:
            board_name = os.path.join(conf.board_dir, conf.name)
            print(f"Saving learning metrics to tensorboard: '{board_name}'")
            self.writer = SummaryWriter(board_name)


    def data_loader(self, data, limit_batches:float, shuffle:bool):
        if limit_batches < 1.0:
            subset_size    = int(len(data) * limit_batches)
            subset_indices = np.random.choice(len(data), size=subset_size, replace=False)
            return DataLoader(data, batch_size=self.conf.batch_size, collate_fn=collate_events, pin_memory=False, num_workers=self.conf.num_data_workers, sampler=SubsetRandomSampler(subset_indices))

        ## using all samples
        return DataLoader(data, batch_size=self.conf.batch_size, shuffle=shuffle, collate_fn=collate_events, pin_memory=False, num_workers=self.conf.num_data_workers)


    def batch_forward(self, b, T):
        h_path = self.model(
            time_uniq  = torch.from_numpy(b["time_uniq"]).to(self.device),
            time_ptr   = torch.from_numpy(b["time_ptr"]).to(self.device),
            X          = torch.from_numpy(b["x"]).to(self.device),
            sample_ids = torch.from_numpy(b["sample_ids"]).to(self.device),
            T          = float(T),
            covs       = torch.from_numpy(b["covs"]).to(self.device),
        )
        return h_path

    def evaluate_binary_dense(self, loader, progress=True):
        self.model.eval()
        logloss_sum   = 0.0
        logloss_count = 0
        time_list     = []
        patient_list  = []
        cols_list     = []
        labels_list   = []
        yhat_list     = []
        obs_occurred_list = []
        num_tasks     = self.conf.output_size

        with torch.no_grad():
            for b in tqdm(loader, leave=False, disable=(progress == False)):
                T = b["time_uniq"][-1] + self.conf.noevent_pad
                yhat   = self.batch_forward(b, T)
                labels = b["labels"].to(self.device)
                mask   = ~torch.isnan(labels)
                Tcount = b["labels"].shape[0]

                tx, px, cols  = torch.where(mask)

                yhat_masked   = yhat[:Tcount][mask]
                labels_masked = labels[mask]

                logloss = self.loss(yhat_masked, labels_masked).sum()

                ## storing data for AUCs
                cols_list.append(cols.cpu())
                labels_list.append(labels_masked.cpu())
                yhat_list.append(yhat_masked.cpu())

                ## for saving trajectories
                time_list.append((self.conf.dt * tx).cpu())
                patient_list.append(b["patient_ids"][px.cpu()])

                time_idx = times_to_idx(b["time"], dt=self.conf.dt)
                ## moving event=1 one step earlier (if at border)
                time_idx[time_idx == mask.shape[0]] -= 1
                obs_occurred_masked = times_to_tensor(
                    torch.LongTensor(time_idx),
                    torch.LongTensor(b["sample_ids"]),
                    shape = mask.shape,
                )[mask]
                obs_occurred_list.append(obs_occurred_masked)

                ## loss
                logloss_sum   += logloss.item()
                logloss_count += yhat_masked.shape[0]

            if len(labels_list) == 0:
                return {
                    "metrics":  compute_metrics([], y_true=[], y_score=[], num_tasks=num_tasks),
                    "logloss":  np.nan,
                    "times":    None,
                    "patients": None,
                    "cols":     None,
                    "labels":   None,
                    "yhat":     None,
                }
            cols   = torch.cat(cols_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            yhat   = torch.cat(yhat_list, dim=0)
            metrics = compute_metrics(cols.numpy(), y_true=labels.numpy(), y_score=yhat.numpy(), num_tasks=num_tasks)

            return {
                'metrics':  metrics,
                'logloss':  logloss_sum / logloss_count,
                "times":    torch.cat(time_list, dim=0),
                "patients": torch.cat(patient_list, dim=0),
                'cols':     cols,
                'labels':   labels,
                'yhat':     torch.sigmoid(yhat),
                "obs_occurred": torch.cat(obs_occurred_list, dim=0),
            }

    def update_board(self, epoch, results_va):
        if self.writer is None:
            return

        self.writer.add_scalar("logloss", results_va["logloss"], epoch)
        metrics_mean = results_va["metrics"].mean()
        for key, val in metrics_mean.items():
            self.writer.add_scalar(key, val, epoch)

    def close_board(self):
        writer.close()
        
    def save_results(self, results_va):
        """
        Saves conf and results_va into a json file.
        Returns the file name.
        """
        out = {}
        out["conf"]       = self.conf.__dict__
        out["results_va"] = {
            "metrics": results_va["metrics"].to_json(),
            "logloss": results_va["logloss"],
        }

        results_file = os.path.join(self.conf.output_dir, f"{self.conf.name}.json")
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)

        with open(results_file, "w") as f:
            json.dump(out, f)
        print(f"Results and conf saved into '{results_file}'.")
        print(f"To load the results:")
        print(f"  import event_gruode as eg")
        print(f'  res = eg.load_results("{results_file}")')
        print(f'  print(res["results_va"]["metrics"].T)   ### printing validation results')
        print()
        return results_file

    def save_yhat(self, results_va):
        """
        Saves cols, labels, and yhat into an .npy file.
        """
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)
        yhat_file = os.path.join(self.conf.output_dir, f"{self.conf.name}-yhat.pt")
        out = {
            'time':    results_va["times"],
            'patient': results_va['patients'],
            'period':  results_va["cols"],
            'label':   results_va["labels"],
            "obs_occurred": results_va["obs_occurred"],
            'yhat':    results_va["yhat"],
        }
        torch.save(out, yhat_file)
        print(f"Saved yhat, cols and labels to '{yhat_file}'.")
    
    def save_model(self):
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)
        model_file = os.path.join(self.conf.output_dir, f"{self.conf.name}.pt")
        torch.save(self.model.state_dict(), model_file)
        print(f"Saved model weights into '{model_file}'.")

    def evaluate_binary(self, loader, progress=True):
        self.model.eval()
        logloss_sum   = 0.0
        logloss_count = 0
        cols_list     = []
        labels_list   = []
        yhat_list     = []
        num_tasks     = self.conf.output_size

        with torch.no_grad():
            for b in tqdm(loader, leave=False, disable=(progress == False)):
                T = b["time_uniq"][-1] + self.conf.noevent_pad
                y_path     = self.batch_forward(b, T)
                labels     = b["labels"].to(self.device)
                sample_ids = torch.from_numpy(b["sample_ids"]).to(self.device)
                tx         = torch.from_numpy(times_to_idx(b["time"], self.conf.dt)).to(self.device)

                labels_at_obs = labels[tx, sample_ids]
                yhat_at_obs   = y_path[tx, sample_ids]
                mask_obs      = ~torch.isnan(labels_at_obs)

                rows, cols    = torch.where(mask_obs)
                labels_masked = labels_at_obs[rows, cols]
                yhat_masked   = yhat_at_obs[rows, cols]

                logloss = self.loss(yhat_masked, labels_masked)
                logloss_sum   += logloss.sum().item()
                logloss_count += logloss.shape[0]

                ## storing data for AUCs
                cols_list.append(cols.cpu())
                labels_list.append(labels_masked.cpu())
                yhat_list.append(yhat_masked.cpu())

            if len(labels_list) == 0:
                return {
                    "metrics": compute_metrics([], y_true=[], y_score=[], num_tasks=num_tasks),
                    "logloss": np.nan,
                }
            cols   = torch.cat(cols_list, dim=0).numpy()
            labels = torch.cat(labels_list, dim=0).numpy()
            yhat   = torch.cat(yhat_list, dim=0).numpy()
            metrics = compute_metrics(cols, y_true=labels, y_score=yhat, num_tasks=num_tasks)

            return {
                'metrics': metrics,
                'logloss': logloss_sum / logloss_count
            }

    def train(self):
        print(f"Train data: {len(self.loader_tr.dataset):8} trajectories")
        print(f"Valid data: {len(self.loader_va.dataset):8} trajectories")
        print(f"Test data:  {len(self.loader_te.dataset):8} trajectories")

        for epoch in range(self.conf.epochs):
            self.model.train()

            loss_means = []
            for b in tqdm(self.loader_tr, leave=False):
                self.optimizer.zero_grad()

                T = b["time_uniq"][-1] + 2.0

                h_path = self.batch_forward(b, T)

                Tcount = b["labels"].shape[0]
                labels = b["labels"].to(self.device)
                mask   = ~torch.isnan(labels)

                output = h_path[:Tcount][mask]
                target = labels[mask]

                losses = self.loss(output, target)
                loss_mean = losses.mean()
                loss_mean.backward()

                self.optimizer.step()

                loss_means.append(loss_mean.detach().item())

            self.scheduler.step()

            ## epoch avg loss
            self.model.eval()
            results_va = self.evaluate_binary_dense(self.loader_va)

            self.update_board(epoch=epoch, results_va=results_va)

            ## TODO: save the model if it is the best for now.
            ## Also save the trajectories
            ## if not best, reduce LR once, next time shutdown

            print(f"-----[ Epoch {epoch} ]-----")
            print(results_va["metrics"].T)
            print()

            intermediate_value = results_va["metrics"]["prec@fpr0.01"].mean()
            if self.trial is not None:
                self.trial.report(intermediate_value, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()


        print("Final results on all tasks on validation set:")
        print(f"  prec@fpr0.01 (va):\n {results_va['metrics']['prec@fpr0.01'].T}")
        print(f"  Logloss (va):      {results_va['logloss']:.5f}")
        print()

        results_file = self.save_results(results_va=results_va)

        if self.conf.save_model:
            self.save_model()
        if self.conf.save_yhat:
            self.save_yhat(results_va=results_va)

        return intermediate_value


def list2str(a):
    return "-".join(str(x) for x in a)

def create_run_single():
    parser = argparse.ArgumentParser(description="Train Event-GRU-Bayes model.")
    parser.add_argument("--sparse_data", help="Data file for sparse data (.pickle)", type=str, default=None)
    parser.add_argument("--data", help="Data file (.cvs)", type=str, default=None)
    parser.add_argument("--covs", help="Covariates file (.csv)", type=str, default=None)
    parser.add_argument("--folds", help="Folding file (.csv)", type=str, default=None, required=True)
    parser.add_argument("--fold_va", help="Validation fold (default 0)", type=int, default=0)
    parser.add_argument("--fold_te", help="Test fold (default 1)", type=int, default=1)
    parser.add_argument("--dt", help="Time step (default 0.1)", type=float, default=0.1)
    parser.add_argument("--ode_slower", help="ODE scale, 10 means ODE step is (dt / 10)", type=float, default=10.0)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--model", help="Batch size", type=str,
        default = "gru_discrete",
        choices = ["gru_discrete", "gru_ode"],
    )
    parser.add_argument("--hidden_size", help="Hidden size", type=int, default=64)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=50)
    parser.add_argument("--output_hidden_size", help="Out hidden size", type=int, default=64)
    parser.add_argument("--cov_hidden_size", help="Covs hidden size", type=int, default=64)
    parser.add_argument("--output_dropout", help="Out hidden size", type=float, default=0.0)
    parser.add_argument("--cov_dropout", help="Covs dropout rate", type=float, default=0.0)
    parser.add_argument("--event_periods", help="Event periods to predict (into future)", nargs="+", default=[1.0, 2.0, 3.0, 4.0, 6.0], type=float)
    parser.add_argument("--add_events_to_x", help="Whether to add events to x matrix", type=int, default=0)
    parser.add_argument("--noevent_pad", help="Padding for final observation", type=float, default=0.0)
    parser.add_argument("--label_buffer", help="Assumes patients are fine for buffer duration (reduces NAs at the end of traj.)", type=float, default=30.0)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--lr_steps", help="Epoch numbers when to drop lr", type=int, nargs="+", default=[40])
    parser.add_argument("--lr_alpha", help="Drop multiplier for lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-5)
    parser.add_argument("--device", help="Device", type=str, default="cuda:0")
    parser.add_argument("--num_data_workers", help="Number of data workers", type=int, default=0)
    parser.add_argument("--save_board", help="Whether to save TensorBoard", type=int, default=1)
    parser.add_argument("--save_model", help="Whether to save model", type=int, default=0)
    parser.add_argument("--save_yhat", help="Whether to save trajectories (yhat, label)", type=int, default=0)
    parser.add_argument("--board_dir", help="Path where to save TensorBoard", type=str, default="boards/")
    parser.add_argument("--output_dir", help="Path where to save conf and model", type=str, default="models/")
    parser.add_argument("--early_stopping_periods", help="Early stopping periods (0 means no stopping)", type=int, default=0)
    parser.add_argument("--limit_train_batches", type=float, help="Use only subset of batches for training", default=1.0)
    parser.add_argument("--limit_val_batches", type=float, help="Use only subset of batches for validation", default=1.0)
    parser.add_argument("--limit_test_batches", type=float, help="Use only subset of batches for test", default=1.0)

    conf = parser.parse_args()
    print(conf)
    return create_run(conf)

def create_run(conf, trial=None):
    if conf.sparse_data is None and conf.data is None:
        parser.print_help()
        raise ValueError("Please specify either '--sparse_data' or '--data'.")
    if conf.sparse_data is not None and conf.data is not None:
        raise ValueError("Both '--sparse_data' and '--data' given. Please specify only one.")

    if conf.model == "gru_ode":
        model = f"gru_ode{conf.ode_slower}"
    else:
        model = "gru_ode0"
    conf.name = f"run_h{conf.hidden_size}-{conf.cov_hidden_size}-{conf.output_hidden_size}_do{conf.cov_dropout:.2f}-{conf.output_dropout:.2f}_epochs{conf.epochs}_lr{conf.lr}_lrsteps{list2str(conf.lr_steps)}_wd{conf.weight_decay:.1e}_dt{conf.dt}_{model}_buf{conf.label_buffer}_ep{list2str(conf.event_periods)}_fold_va{conf.fold_va}_fold_te{conf.fold_te}"

    if conf.data is not None:
        data = pd.read_csv(conf.data)
        assert "patient_id" in data.columns, "data must have 'patient_id' column"
        assert "time" in data.columns, "data must have 'time' column"
        assert "event" in data.columns, "data must have 'event' column"
    elif conf.sparse_data is not None:
        data = pickle.load(open(conf.sparse_data, "rb"))

    folds = pd.read_csv(conf.folds)

    covs = None
    if conf.covs is not None:
        covs = pd.read_csv(conf.covs)

    run = EventRun(data, covs=covs, folds=folds, conf=conf, trial=trial)
    return run

if __name__ == "__main__":
    run()
