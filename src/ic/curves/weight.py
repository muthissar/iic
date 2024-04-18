import numpy as np
import torch


from dataclasses import dataclass
from typing import Callable, List, Union


@dataclass
class Weight(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


@dataclass
class MovingAverage(Weight):
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
        # NOTE: (bz, observations, channels, eval_points)
        # NOTE: numerical stability for 0 * inf
        e = 1e-9
        mov_avg = self.cw*(-self.c_*(time_diffs+e)).exp()
        mov_avg[time_diffs>=self.window_size] = 0
        return mov_avg


@dataclass
class Hann(Weight):
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
        # NOTE: (bz, observations, channels, eval_points)
        if isinstance(self.channel_weight, Callable):
            cw = self.cw(torch.arange(torch.prod(torch.tensor(time_diffs.shape[1:3])))).view(*time_diffs.shape[1:3])[None,...,None]
        else:
            cw = self.cw    
        weight = cw*(0.5 +.5*np.cos(np.pi*time_diffs/self.window_size)) # /self.window_size
        weight[time_diffs>=self.window_size] = 0
        return weight