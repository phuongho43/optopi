# optopi

Modeling light inducible protein interaction systems via ODEs.

## Getting Started

### 1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/)

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Run Code

Fit model to [ddFP](https://doi.org/10.1016/j.chembiol.2012.01.006) data for [LOVTRAP](https://doi.org/10.1038/nmeth.3926) and [iLID](https://doi.org/10.1073/pnas.1417910112).

```bash
$ uv run fit_model
```

Simulate the response of the composite system RA-LOVfast + Zdk-iLIDslow + B3-sspBn to light stimulation.

```bash
$ uv run sim_model
```

## Workflow

### 1. ddFP Biosensor Images

| RA-LOVfast + B3-Zdk |
| Before Stimulation | After Stimulation|
|:-------------------------:|:-------------------------:|
| ![](/example/data/LOV/I427V/60.1.png) | ![](/example/data/LOV/I427V/61.9.png) |

### 2. ODE Model Fit

|                     LOVfast                     |                     LOVslow                     |
| :---------------------------------------------: | :---------------------------------------------: |
| ![](/example/sim_model/LOV/I427V/model-fit.png) | ![](/example/sim_model/LOV/V416I/model-fit.png) |

|                     LIDfast                     |                     LIDslow                     |
| :---------------------------------------------: | :---------------------------------------------: |
| ![](/example/sim_model/LID/I427V/model-fit.png) | ![](/example/sim_model/LID/V416I/model-fit.png) |

### 3. Sparse Decoder Response Prediction

<p align="left">
    <img src="/example/sim_model/sparse_decoder/prediction.png" width="50%" />
</p>

### 4. FM Response Prediction

<p align="left">
    <img src="/example/sim_model/fm-response.png" width="50%" />
</p>
