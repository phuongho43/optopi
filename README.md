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

<p align="center">
    <figure>
        <img src="/example/data/LOV/I427V/60.1.png" width="50%" />
        <figurecaption>LOVfast [before light]</figurecaption>
    </figure>
    <figure>
        <img src="/example/data/LOV/I427V/61.9.png" width="50%" />
        <figurecaption>LOVfast [after light]</figurecaption>
    </figure>
</p>
<p align="center">
    <figure>
        <img src="/example/data/LID/I427V/60.1.png" width="50%" />
        <figurecaption>LIDfast [before light]</figurecaption>
    </figure>
    <figure>
        <img src="/example/data/LID/I427V/62.0.png" width="50%" />
        <figurecaption>LIDfast [after light]</figurecaption>
    </figure>
</p>

### 2. Binding Dynamics

<p align="center">
    <figure>
        <img src="/example/data/LOV/I427V/uy.png" width="50%" />
        <figurecaption>LOVfast</figurecaption>
    </figure>
    <figure>
        <img src="/example/data/LOV/V416I/uy.png" width="50%" />
        <figurecaption>LOVslow</figurecaption>
    </figure>
</p>
<p align="center">
    <figure>
        <img src="/example/data/LID/I427V/uy.png" width="50%" />
        <figurecaption>LIDfast</figurecaption>
    </figure>
    <figure>
        <img src="/example/data/LID/V416I/uy.png" width="50%" />
        <figurecaption>LIDslow</figurecaption>
    </figure>
</p>

### 3. ODE Model Fit

<p align="center">
    <figure>
        <img src="/example/sim_model/LOV/I427V/model-fit.png" width="50%" />
        <figurecaption>LOVfast</figurecaption>
    </figure>
    <figure>
        <img src="/example/sim_model/LOV/V416I/model-fit.png" width="50%" />
        <figurecaption>LOVslow</figurecaption>
    </figure>
</p>
<p align="center">
    <figure>
        <img src="/example/sim_model/LID/I427V/model-fit.png" width="50%" />
        <figurecaption>LIDfast</figurecaption>
    </figure>
    <figure>
        <img src="/example/sim_model/LID/V416I/model-fit.png" width="50%" />
        <figurecaption>LIDslow</figurecaption>
    </figure>
</p>

### 4. Sparse Decoder Response Prediction

<p align="center">
    <img src="/example/sim_model/sparse_decoder/prediction.png" width="50%" />
</p>

### 5. FM Response Prediction

<p align="center">
    <img src="/example/sim_model/fm-response.png" width="50%" />
</p>
