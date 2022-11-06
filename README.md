
Code for "Efficient and Stable Off-policy Training via Behavior-aware Evolutionary Learning" accepted by CoRL2022.

# Usage
It is straight forward to run BEL in a local conda environment. First make sure you have `mujoco200` and `mjkey.txt` in your `~/.mujoco`. Then create the conda env and run `bel.py`. By default, experiment's logs will be output to a directory called `results`.

```
conda env create -f environment.yml
conda activate bel
python bel.py --env_name Ant-v3 --seed 10
```