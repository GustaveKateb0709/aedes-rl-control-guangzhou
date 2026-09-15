# Trained policies

The trained policies and their observation-normalisation statistics are **not**
stored in this repository: the full set is about 1.4 GB, dominated by the
flattened-full-space arms whose action head has 59,049 outputs.

They are distributed with the archived release (see the Data availability
section of the top-level README) and can be regenerated from scratch with the
fixed seeds in `code/`, e.g.

```bash
cd code
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_cinf_s0
python k_confound.py --mode train --workers 3     # the action-space control arms
python loyo.py --mode train --workers 3           # the cross-validation folds
python capacity_sweep.py --mode train --workers 3 --only ppo
```

Model filenames follow `models/ppo_<config>_s<seed>.zip` with a matching
`_vecnorm.pkl`; the configuration tags are documented in the top-level README.
