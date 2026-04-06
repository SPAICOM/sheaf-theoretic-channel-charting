# A Sheaf-Theoretic Framework for Distributed Multi-Site Channel Charting

<h5 align="center">
     
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/sheaf-theoretic-channel-charting/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Channel charting (CC) enables data-driven user localization in wireless networks by embedding channel state information (CSI) into low-dimensional representations. In multi-cell scenarios, each base station independently learns a local chart via neural encoders, leading to misaligned representation spaces across overlapping coverage areas. This lack of consistency hinders network-level tasks such as user tracking, handover prediction, and resource allocation. To address this issue, we propose a principled framework for multi-site channel charting based on topological signal processing. We model the collection of local charts as a network sheaf, which encodes consistency constraints across the network and enables the coherent integration of locally learned representations into a shared global structure. This formulation introduces an interpretable inductive bias that promotes alignment across base stations while preserving local geometric fidelity. Building on this model, we develop a multi-site channel charting architecture and an alternating optimization algorithm that jointly updates neural encoders and inter-site orthogonal transport maps, with theoretical guarantees on consistency. Experimental results validate the effectiveness of the proposed approach, demonstrating improved cross-site alignment without degrading the quality of local embeddings.

## Simulations

This section provides the necessary commands to reproduce the experiments presented in the paper. The project uses [`just`](https://github.com/casey/just) as a task runner — refer to the [Dependencies](#dependencies) section for setup instructions.

### Run all experiments

To reproduce all paper results in one shot, run both the proposed sheaf-based methods and all baselines:

```bash
# Run all sheaf-based orchestrators
just sheaf

# Run all baselines
just baselines
```

Alternatively, to run everything at once:

```bash
just sim-all
```

### Proposed methods

The following recipes run the individual sheaf-based orchestrators proposed in the paper:

```bash
just fb       # Flat Bundle
just bun      # Bundle
just cs       # Cover Sheaf
just ds       # Diagonal Sheaf
just nds      # Neural Diagonal Sheaf
```

### Baselines

```bash
just fed      # Federated Channel Charting
just pfed     # Personalized Federated Channel Charting
just ot       # Optimal Transport Channel Charting
just vanilla  # Vanilla (baseline siamese network)
```

### Single-agent training

To run single-agent training without any orchestrator:

```bash
just single
```

### Evaluation

Once the simulations have completed, evaluate all orchestrators and save metrics to `results/eval_metrics.parquet`:

```bash
just eval
```

### Plotting

To visualize dataset trajectories:

```bash
just plot
```

To visualize latent representations from saved evaluation results:

```bash
just viz
```

To plot the results table (FOSCTTM, KS) and CT/TW vs K line plots:

```bash
just results
```

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://github.com/astral-sh/uv#installation)
- [`just`](https://github.com/casey/just#installation)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use. 🚀

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Authors

- [Enrico Grimaldi](https://scholar.google.com/citations?user=Y-31eCwAAAAJ)
- [Leonardo Di Nino](https://scholar.google.com/citations?user=4UdFEvAAAAAJ)
- [Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ)
- [Gabriele D'Acunto](https://scholar.google.com/citations?user=dIVgmlUAAAAJ)
- [Sergio Barbarossa](https://scholar.google.com/citations?user=2woHFu8AAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?user=VZYvspQAAAAJ)

## Used Technologies


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-89CFF0?style=for-the-badge&logo=hyperland&logoColor=white)
![w&b](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
