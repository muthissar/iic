import numpy as np
import torch


from dataclasses import dataclass
from typing import Callable, List, Union


@dataclass
class Weight(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    """
    Abstract base class for IIC weight function (lambda).

    It is a callable that takes a tensor of time differences and return a weighting. 

    Methods
    -------
    __call__(time_diffs : torch.FloatTensor) -> torch.FloatTensor
        Abstract method for calculating weights.
        This method should be implemented by subclasses.

    Parameters
    ----------
    time_diffs : torch.FloatTensor
        A tensor of time differences for which to calculate weights.
    """
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        """
        Abstract method for calculating weights.

        Parameters
        ----------
        time_diffs : torch.FloatTensor
            A tensor of time differences for which to calculate weights.

        Returns
        -------
        torch.FloatTensor
            A tensor of weights corresponding to the input time differences.
        """
        raise NotImplementedError


@dataclass
class MovingAverage(Weight):
    """
    A class used to represent a Moving Average weight function.

    This class inherits from the Weight class and implements the __call__ method
    to calculate weights based on a moving average.

    Attributes
    ----------
    window_size : float
        The size of the window for the moving average.
    decay : Union[float, List[Union[float, str]]]
        The decay factor(s) for the moving average.
    channel_weight : Union[float, List[Union[float, str]]], optional
        The weight(s) for each channel (default is 1.0).

    Methods
    -------
    __post_init__():
        Converts decay and channel_weight to lists of floats and initializes tensors.
    __call__(time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        Calculates weights based on the moving average of time differences.
    """
    window_size : float
    decay: Union[float, List[Union[float, str]]]
    channel_weight: Union[float, List[Union[float, str]]] = 1.
    def __post_init__(self):
        if isinstance(self.decay, List):
            self.decay = [float(c_) for c_ in self.decay]
        elif isinstance(self.decay, float):
            self.decay = [self.decay]
        if isinstance(self.channel_weight, List):
            self.channel_weight = [float(a) for a in self.channel_weight]
        elif isinstance(self.channel_weight, float):
            self.channel_weight = [self.channel_weight]
        self.c_ = torch.tensor(self.decay)[None, None, :, None] # bz=1, tok=1, channels
        self.cw = torch.tensor(self.channel_weight)[None, None, :, None] # bz=1, tok=1, channels
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates weights based on the moving average of time differences.

        Parameters
        ----------
        time_diffs : torch.FloatTensor
            A tensor of time differences for which to calculate weights.

        Returns
        -------
        torch.FloatTensor
            A tensor of weights corresponding to the moving average of the input time differences.
        """
        e = 1e-9
        mov_avg = self.cw*(-self.c_*(time_diffs+e)).exp()
        mov_avg[time_diffs>=self.window_size] = 0
        return mov_avg


@dataclass
class Hann(Weight):
    """
    A class used to represent a Hann window function as a weight function.

    This class inherits from the Weight class and implements the __call__ method
    to calculate weights based on a Hann window function.

    Attributes
    ----------
    window_size : float
        The size of the window for the Hann function.
    channel_weight : Union[float, List[Union[float, str]], Callable[[torch.LongTensor],torch.FloatTensor]], optional
        The weight(s) for each channel (default is 1.0).

    Methods
    -------
    __post_init__():
        Converts channel_weight to a tensor and initializes it.
    __call__(time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        Calculates weights based on the Hann window function of time differences.
    """
    window_size : float
    channel_weight: Union[float, List[Union[float, str]], Callable[[torch.LongTensor],torch.FloatTensor]] = 1.
    def __post_init__(self):
        if isinstance(self.channel_weight, Callable):
            self.cw = self.channel_weight
        else:
            if isinstance(self.channel_weight, float):
                self.channel_weight = [self.channel_weight]
            self.cw = torch.tensor(self.channel_weight)[None, None, :, None] # bz=1, tok=1, channels
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates weights based on the Hann window function of time differences.

        Parameters
        ----------
        time_diffs : torch.FloatTensor
            A tensor of time differences for which to calculate weights.

        Returns
        -------
        torch.FloatTensor
            A tensor of weights corresponding to the Hann window function of the input time differences.
        """
        # NOTE: (bz, observations, channels, eval_points)
        if isinstance(self.channel_weight, Callable):
            cw = self.cw(torch.arange(torch.prod(torch.tensor(time_diffs.shape[1:3])))).view(*time_diffs.shape[1:3])[None,...,None]
        else:
            cw = self.cw    
        weight = cw*(0.5 +.5*np.cos(np.pi*time_diffs/self.window_size))
        weight[time_diffs>=self.window_size] = 0
        return weight