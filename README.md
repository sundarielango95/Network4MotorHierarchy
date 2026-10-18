# Lateralised motor network for visuomotor adaptation

Code for the paper:

> Elango S, Chakravarthy VS, Mutha PK (2024). **A lateralized motor network in order to understand adaptation to visuomotor rotation.** *Journal of Neural Engineering* 21(3). [doi:10.1088/1741-2552/ad4211](https://doi.org/10.1088/1741-2552/ad4211) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/38653251/)

## What the model does

A hierarchical network controls a two-joint arm reaching to targets in 2D:

- a learned forward model of the arm (the **plant**),
- a motor-cortex (**M1**) controller that drives it,
- left and right posterior parietal cortex (**PPC**) networks that feed M1. The left PPC learns movement direction and the right PPC learns movement distance (gain).

The model is trained under rotated visual feedback (visuomotor rotation), then tested on the untrained arm and after simulated parietal stimulation. It reproduces experimental findings on how adaptation transfers between arms, and separates what the left and right parietal areas compute in unimanual and bimanual movement.

## Layout

| Path | Contents |
|---|---|
| `models/model_architecture.py` | Every network class: plant, M1 controller, left and right PPC, and variants |
| `models/` | Trained weights for some subjects (`plant_xandy_*`, `m1_dist_*`), saved with `torch.save` |
| `training/plant_train.py` | Trains the forward model of the arm |
| `training/m1.py` | Trains the M1 controller |
| `training/train_lppc.py`, `training/train_rppc.py` | Train the left and right PPC networks |
| `testing/plant_test.py` | Tests the forward model |
| `testing/m1_ppc_test.py`, `testing/pm_test.py` | Test the trained controller across subjects and targets |
| `testing/test_after_rot_train.py` | Tests after visuomotor-rotation training |
| `testing/test_after_stim.py` | Tests after simulated parietal stimulation |
| `testing/test_rppc_lppc.py` | Compares the left and right PPC under normal, rotated and stimulated conditions |
| `testing/test_extensions.py` | Further experiments |
| `dataset/second_order_ode.m` | MATLAB script that generates the arm trajectories used for training |

## Running it

Run in this order: generate the data, then `plant_train.py` → `m1.py` → `train_lppc.py` and `train_rppc.py` → the testing scripts.

This is the research code as it was run for the paper. Before running:

1. **Data.** The training data is not in the repository. Run `dataset/second_order_ode.m` in MATLAB to generate it. The training scripts read `data/train_random_theta_vel_dist.xlsx`.
2. **Model paths.** The scripts load weights from subfolders such as `models/plant/` and `models/rppc_lppc/m1/`, but the weights in this repository sit directly in `models/`. Move the files or edit the paths.
3. **Imports.** The scripts import `model_architecture`, so run them from the repository root with `models/` on the Python path:
   ```bash
   pip install -r requirements.txt
   PYTHONPATH=models python training/plant_train.py
   ```
4. **`test_extensions.py`** imports a module called `brain`. The `brain` class is in `models/model_architecture.py`, so change that import before running it.

## Citation

```bibtex
@article{elango2024lateralized,
  title   = {A lateralized motor network in order to understand adaptation to visuomotor rotation},
  author  = {Elango, Sundari and Chakravarthy, V. Srinivasa and Mutha, Pratik K.},
  journal = {Journal of Neural Engineering},
  volume  = {21},
  number  = {3},
  year    = {2024},
  doi     = {10.1088/1741-2552/ad4211}
}
```

## Contact

Sundari Elango, sundarielango95@gmail.com
