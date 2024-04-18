# IIC
Code for computing _Instantaneous Information Content (IIC)_, as described in [Controlling Surprisal in Music Generation via Information Content Curve Matching]().
## Install
```bash
pip install .
```

## Demo
A Jupyter notebook demo is presented in [IIC.ipynb](./IIC.ipynb), showing how to compute IICs using [MidiTok](https://github.com/Natooz/MidiTok)'s implementation of the [Structured MIDI Encoding](https://arxiv.org/pdf/2107.05944.pdf) with token Information Content precomputed by a [PIA model](https://arxiv.org/pdf/2107.05944.pdf). 

To run the demo, install the package with:
```bash
pip install "./[demo]"
```
