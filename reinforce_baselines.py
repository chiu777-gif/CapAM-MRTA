import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
import csv
from pathlib import Path
from problems.mrta.config import MRTAConfig, split_seed
from problems.mrta.data import collate_mrta
from train import rollout, get_inner_model

class Baseline(object):

    collate_fn = staticmethod(collate_mrta)

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha == 1:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha == 1:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = min(1., (epoch + 1) / float(self.n_epochs))
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def collate_fn(self, batch):
        return (self.baseline if self.alpha == 1 else self.warmup_baseline).collate_fn(batch)

    def state_dict(self):
        return {'baseline': self.baseline.state_dict(),
                'warmup': self.warmup_baseline.state_dict(), 'alpha': self.alpha}

    def load_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict['baseline'])
        self.warmup_baseline.load_state_dict(state_dict['warmup'])
        self.alpha = state_dict['alpha']


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v.to(c.device) + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


def collate_baseline(batch):
    return {'data': collate_mrta([item['data'] for item in batch]),
            'baseline': torch.stack([item['baseline'] for item in batch])}


class RolloutBaseline(Baseline):
    collate_fn = staticmethod(collate_baseline)


    def __init__(self, model, problem, opts, epoch=0):
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        if dataset is None:
            self.dataset = self.problem.make_training_dataset(
                MRTAConfig.from_options(self.opts), self.opts.val_size,
                split_seed(self.opts.seed, 'baseline', epoch))
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        p_val = None
        updated = False
        if candidate_mean < self.mean and len(candidate_vals) > 1:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2
            if t < 0 and p_val < self.opts.bl_alpha:
                self._update_model(model, epoch)
                updated = True
        path = Path(self.opts.save_dir) / 'convergence.csv'
        exists = path.exists()
        with path.open('a', newline='') as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(['epoch', 'candidate_cost', 'baseline_cost', 'p_value', 'updated'])
            writer.writerow([epoch, candidate_mean, self.mean, p_val, updated])

    def state_dict(self):
        return {
            'model': get_inner_model(self.model).state_dict(),
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        saved_model = state_dict['model']
        weights = saved_model if isinstance(saved_model, dict) else get_inner_model(saved_model).state_dict()
        get_inner_model(load_model).load_state_dict(weights)
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
