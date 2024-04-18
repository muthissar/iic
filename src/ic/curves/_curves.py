import einops
import numpy as np
import torch


from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional


@dataclass
class ICCurve(Callable[[torch.FloatTensor], torch.FloatTensor]):
    """
    A class used to represent an Instantaneous Information Content Curve (IIC).

    This class is a callable that takes a tensor of times and returns a tensor of corresponding IICs.

    Methods
    -------
    __call__(t : torch.FloatTensor) -> torch.FloatTensor:
        Abstract method for calculating intensities.
        This method should be implemented by subclasses.

    Parameters
    ----------
    t : torch.FloatTensor
        A tensor of times for which to calculate intensities.
    """
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


@dataclass
class DrawnICCurve(ICCurve):
    relative_time: bool
    def set_placeholder_length(self, placholder_length : float):
        if self.relative_time:
            self._placeholder_scale = 1 / placholder_length
    def __post_init__(self):
        self._placeholder_scale = 1.


@dataclass
class LinearInterpolation(DrawnICCurve):
    timepoints: List[float]
    ics: List[List[float]]
    def __post_init__(self):
        super().__post_init__()
        assert len(self.timepoints) == len(self.ics)
        self._timepoints = np.array(self.timepoints)
        self._ics = np.array(self.ics)
    def __call__(self, t : List[torch.FloatTensor]) -> List[torch.FloatTensor]:
        # t : bz, T 
        lens = [len(tt) for tt in t]
        t = torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=-1)
        assert t.dim() == 2
        orig_shape  = t.shape
        t = (t* self._placeholder_scale).numpy()
        res = torch.FloatTensor(np.stack([np.interp(t.flatten(), self._timepoints, ics) for ics in self._ics.T], axis=-1))
        res = res.view(*orig_shape, -1)
        res = [r[:l] for r, l in zip(res, lens)]
        return res

@dataclass
class Piecewise(DrawnICCurve):
    # NOTE alternatively scipy.interpolate.interp1d(x, y, kind='nearest'), but it's deprecated
    timepoints: List[float]
    ics: List[List[float]]
    def __post_init__(self):
        super().__post_init__()
        assert len(self.timepoints) == len(self.ics) - 1
        self._timepoints = np.array(self.timepoints)
        self._ics = np.array(self.ics)
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        # Define the conditions for each step
        t = (t* self._placeholder_scale).numpy()
        timepoints = self._timepoints
        conditions = [t < timepoints[0]]
        for i in range(len(timepoints) - 1):
            conditions.append((t >= timepoints[i]) & (t < timepoints[i+1]))
        conditions.append(t >= timepoints[-1])
        return torch.tensor(np.stack([np.piecewise(t, conditions, self._ics[:,i]) for i in range(self._ics.shape[1])],axis=-1)[None])


@dataclass
class Interpolator(ICCurve):
    metric_times : Iterable[torch.FloatTensor]
    metric: Iterable[torch.FloatTensor]
    weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
    metric_clip:  Optional[torch.FloatTensor]= None
    reduce_equal_times: str = 'sum' # 'max', 'mean'
    """
    A class used to interpolate between different metrics over time.

    This class inherits from the ICCurve class and implements the __call__ method
    to calculate interpolated values based on a given weight function.

    Attributes
    ----------
    metric_times : Iterable[torch.FloatTensor]
        The times at which the metrics are measured.
    metric: Iterable[torch.FloatTensor]
        The metrics to be interpolated.
    weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
        The weight function to be used for interpolation.
    metric_clip: Optional[torch.FloatTensor]
        The maximum value for the metrics (default is None).
    reduce_equal_times: str
        The method to be used for reducing metrics at the same time (default is 'sum').

    Methods
    -------
    __post_init__():
        Validates the input and pads the metric_times and metric sequences.
    __call__(t : torch.FloatTensor) -> torch.FloatTensor:
        Calculates interpolated values based on the weight function and the metrics.
    """
    def __post_init__(self):
        lens = [len(ic) for ic in self.metric]
        assert all(l == len(ic) for l, ic in zip(lens, self.metric))
        self.metric_times = torch.nn.utils.rnn.pad_sequence(self.metric_times, batch_first=True)
        self.metric = torch.nn.utils.rnn.pad_sequence(self.metric, batch_first=True)
        if self.reduce_equal_times in ['max', 'mean']:
            times_t = einops.rearrange(self.metric_times, 'bz tok chan -> (bz chan) tok')
            ics_t = einops.rearrange(self.metric, 'bz tok chan -> (bz chan) tok')
            for times, ics in zip(times_t, ics_t):
                unique, inverse, count = times.unique( return_inverse=True, return_counts=True)
                for t,c in zip(unique, count):
                    time_mask = times==t
                    max_val, max_idx = ics[time_mask].max(dim=0)
                    # max
                    if self.reduce_equal_times == 'max':
                        ics[time_mask] = max_val/c
                    # mean
                    else:
                        ics[time_mask] /= c
        elif self.reduce_equal_times != 'sum':
            raise NotImplementedError

        self.metric_times = self.metric_times[..., None]
        if self.metric_clip is not  None:
            self.metric = torch.where(self.metric > self.metric_clip, self.metric_clip, self.metric)
        assert self.metric_times.dim() == 4 and self.metric.dim() == 3 # bz, tokens, channels, (t=1?)

    def __call__(self, t : List[torch.FloatTensor]) -> List[torch.FloatTensor]:
        """
        Calculates interpolated values based on the weight function and the metrics.

        Parameters
        ----------
        t : torch.FloatTensor
            The times at which to calculate the interpolated values.

        Returns
        -------
        torch.FloatTensor
            The interpolated values at the input times.
        """
        lens = [len(tt) for tt in t]
        t = torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=-1)
        assert t.dim() == 2 # bz, t
        time_diffs = t[:, None, None] - self.metric_times
        w = self.weight_fn(time_diffs)
        w[time_diffs <.0] = 0.
        # NOTE: the ic padding cancels automatically.

        metric = self.metric.expand(w.shape[0], *self.metric.shape[1:])
        ret = einops.einsum(w, metric, 'bz tok chan t, bz tok chan -> bz t chan')
        return [r[:l] for r, l in zip(ret, lens)]