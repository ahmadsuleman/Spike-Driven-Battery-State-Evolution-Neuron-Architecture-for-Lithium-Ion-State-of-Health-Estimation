# DABSEN: Degradation-Aware Spike-Driven Battery State Evolution Neuron

This repository contains the implementation and experimental pipeline for **DABSEN**, a degradation-aware recurrent spiking architecture for lithium-ion battery state-of-health (SOH) estimation from completed discharge-cycle data.

DABSEN estimates cycle-level SOH using voltage, current, and temperature descriptors from completed discharge cycles. Capacity labels, SOH labels, SOC labels, cycle-index features, future-cycle information, and capacity-derived engineered descriptors are excluded from the model input.

## Overview

The proposed model introduces a spike-driven recurrent neuron with:

- window-local degradation-inspired latent evidence,
- recovery-relaxation dynamics,
- membrane-potential evolution,
- spike-trace memory,
- sign-constrained degradation readout.

The evaluation uses an expanded leave-one-battery-out protocol on processed NASA lithium-ion battery aging data.

The exact folder names may be adjusted depending on the local setup.

## Environment

Experiments were run with:

- Python 3.10.20
- PyTorch 2.4.1+cu118
- CUDA 11.8
- NVIDIA GeForce RTX 2080 GPU

The implementation also uses NumPy, pandas, scikit-learn, SciPy, tqdm, and Matplotlib.

## Data Preparation

The processed dataset is generated from the `cleaned_dataset` directory and saved as:

```text
battery_health_dataset.csv
```

Preprocessing steps:

1. Use completed discharge cycles only.
2. Truncate each discharge trace before the first voltage sample below `2.7 V`.
3. Remove cycles with fewer than 20 post-truncation samples.
4. Remove cycles with completed-discharge capacity below `1.4 Ah`.
5. Compute capacity labels by Coulomb counting after cutoff truncation.
6. Normalize SOH using nominal capacity `Q_nom = 2.0 Ah`.
7. Extract 75 features per cycle from voltage, current, and temperature:
   - 20 sequential sample-count bin means per signal,
   - mean,
   - standard deviation,
   - start value,
   - end value,
   - end-minus-start delta.

## Evaluation Protocol

The experiments use an expanded leave-one-battery-out protocol.

For each fold:

- one complete battery is held out for testing,
- four batteries are used for validation,
- the remaining batteries are used for training.

The random seeds are:

```text
42, 12, 88
```

Feature normalization is fitted only on the training batteries in each fold and then applied unchanged to validation and test batteries.

Batteries B0049-B0052 are excluded before analysis. Batteries with fewer than eight valid `L = 12` windows are excluded from formal held-out testing.

## Training Configuration

Main settings:

- Input window length: `L = 12`
- Input dimension: `F = 75`
- Batch size: `64`
- Maximum epochs: `500`
- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-5`
- Gradient clipping: `1.0`
- Checkpoint rule: lowest validation RMSE
- DABSEN hidden dimension: `74`
- Dropout: `0.10`

DABSEN-Core uses endpoint-only MSE supervision. DABSEN-Reg adds optional label-free trajectory and spike-activity penalties.

## Running the Experiments

Open and run the main notebook:

```text
PINT_BSEN_Q1_RIGOROUS_EXPERIMENTS_15_CHANGES.ipynb
```

Recommended execution order:

1. environment and imports,
2. dataset preprocessing,
3. split construction,
4. feature normalization,
5. model training,
6. baseline training,
7. metric calculation,
8. statistical analysis,
9. figure and table generation.

## Outputs

The experiments report:

- RMSE,
- MAE,
- MAPE,
- pooled R²,
- signed bias,
- final-third MAE,
- EOL threshold metrics,
- ablation metrics,
- spike rate,
- estimated SynOps.

End-of-life analysis is evaluated at the `80%` SOH threshold.

## Reproducibility Notes

For each seed and fold, Python `random`, NumPy, and PyTorch seeds are reset. CUDA seeds are also set using:

```python
torch.cuda.manual_seed_all(seed)
```

cuDNN deterministic mode is disabled and cuDNN benchmarking is enabled.

All neural baselines use the same:

- input windows,
- train/validation/test battery splits,
- normalization rule,
- random seeds,
- endpoint target convention,
- validation-checkpoint rule.

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{dabsen_soh,
  title={Degradation-Aware Spike-Driven Battery State Evolution Neuron Architecture for Lithium-Ion State-of-Health Estimation},
  author={Suleman, Ahmad and Khan, Misha Urooj and Adarbah, Haitham},
  journal={IEEE Access},
  year={2026}
}
```

Update the BibTeX entry after publication with the final volume, pages, DOI, and publication year.

## License

Add the intended license for this repository, for example:

```text
MIT License
```

or replace this section with the license required by the project or institution.

## Contact

For questions or issues, contact the corresponding author listed in the manuscript.
