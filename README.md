# Compute the Curvature of the input of a Neural Network

This repository contains the code used for the article *Cartan moving frames and the data manifolds* by Eliot Tron, Rita Fioresi, Nicolas Couellan and Stéphane Puechmorel, available at [https://doi.org/10.1007/s41884-024-00159-8](https://doi.org/10.1007/s41884-024-00159-8)

# How to use CurvComputeNN
## Prerequisites
The code has been written under python >3.9. 

## [main.py](./main.py)
```
usage: main.py [-h] [--datasets name [name ...]] [--restrict class] [--nsample N]
               [--task {curvature,foliation,rank2D,proba2D,trace2D,gradproba,connection-forms}]
               [--random] [--seed seed] [--savedirectory path] [--double] [--maxpool] [--cpu]
               [--nl f [f ...]]

CurvComputeNN: Compute the connection and curvature forms of the DIM associated to a neural
network.

options:
  -h, --help            show this help message and exit
  --datasets name [name ...]
                        Dataset name to be used.
  --restrict class      Class to restrict the main dataset to if needed.
  --nsample N           Number of initial points to consider.
  --task {curvature,foliation,rank2D,proba2D,trace2D,gradproba,connection-forms}
                        Task.
  --random              Permutes randomly the inputs.
  --seed seed           Seed to use if not random.
  --savedirectory path  Path to the directory to save the outputs in.
  --double              Use double precision (1e-16) for the computations (recommended).
  --maxpool             Use the legacy architecture with maxpool2D instead of avgpool2d.
  --cpu                 Force device to cpu.
  --nl f [f ...]        Non linearity used by the network.
```

### Task:
- `curvature`: compute the curvature forms.
- `foliation`: compute and plot the data foliation.
- `rank2D`: plot the rank of the DIM projected onto two input dimensions.
- `proba2D`: plot the max probability density predicted by the network projected onto two input dimensions.
- `trace2D`: plot the trace of the DIM projected onto two input dimensions.
- `gradproba`: compute the Jacobian matrix of the network.
- `connection-forms`: compute the connection forms.
