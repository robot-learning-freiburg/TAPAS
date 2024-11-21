# The Art of Imitation: Learning Long-Horizon Manipulation Tasks from Few Demonstrations
Repository providing the source code for the paper "The Art of Imitation: Learning Long-Horizon Manipulation Tasks from Few Demonstrations", see the [project website](http://tapas-gmm.cs.uni-freiburg.de/) and [video](https://youtu.be/3BilJXwdXLI).

Please cite the paper as follows:

```
@ARTICLE{vonhartz2024art,
  author={von Hartz, Jan Ole and Welschehold, Tim and Valada, Abhinav and Boedecker, Joschka},
  journal={IEEE Robotics and Automation Letters}, 
  title={The Art of Imitation: Learning Long-Horizon Manipulation Tasks From Few Demonstrations}, 
  year={2024},
  volume={9},
  number={12},
  pages={11369-11376},
  doi={10.1109/LRA.2024.3487506}}

```

# License
For academic usage, the code is released under the GPLv3 license. For any commercial purpose, please contact the authors.

This release further builds on code released earlier projects, including
- BASK: https://github.com/robot-learning-freiburg/bask
- DON: https://github.com/RobotLocomotion/pytorch-dense-correspondence
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
- riepybdlib: https://github.com/vonHartz/riepybdlib

Please also consider citing these, if you use relevant parts of the code.


# Installation
## Python
Developed with Python 3.10. Lower version will not work because of new syntax elements.


## Packages
- Create a Python env (using Conda) and install the latest Pytorch version.

- Install riepybdlib:
 ```
 git clone git@github.com:vonHartz/riepybdlib.git
 cd riepybdlib
 pip install -e .
 ```

- Install the Simulator/Environment (RLBench, ManiSkill, real Franka Emika) of your choosing (see below for instructions). Make sure to set up a separate conda env for each to avoid package conflicts. The setup.py specifies extra requirements for each option (rlbench, maniskill, franka). (There's also extras for the diffusion policy.)

- Install the remaining requirements and this package via
```
pip install -r requirements.txt
pip install .
```

## Simulators/Environments

### ManiSkill2:
Install our fork from https://github.com/vonHartz/ManiSkill2
 ```
 git clone git@github.com:vonHartz/ManiSkill2.git
 cd ManiSkill2s
pip install -r requirements.txt
 pip install -e .
 ```

### RLBench
Install our fork form https://github.com/vonHartz/RLBench by following the instructions in that repo.
After activating your RLBench conda env, run `source rlbench_mode.sh` to set up Coppelia and QT paths.


### Franka Emika
The environment file is specific to our robot setup. You might still find it useful for inspiration.


## Motion Planner
(Required for getting demos in ManiSkill2 envs and for post-processing with TOPPRA.)
If you need the motion planner policy, install [pinochio](https://stack-of-tasks.github.io/pinocchio/download.html) via:
<!-- ```
git clone git@github.com:haosulab/MPlib.git
cd MPLib
git submodule update --init --recursive
``` -->

```
sudo apt-get install robotpkg-py310-pinocchio
```

Add to `.bashrc`:
```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH # Adapt your desired python version here
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```

Try `pip install mplib`.

If no package for your Python version exists yet, you can try to download the appropriate release of MPLib from https://github.com/vonHartz/MPlib/releases and install it via `pip install <filename>`.
If it does not exist there either, try building the wheel in the container.


Note, that the official distribution of mplib does currently not expose the SD (desired duration) variant of TOPPRA in the planner class.
I've opened an issue here https://github.com/haosulab/MPlib/issues/91 and will update the instructions as soon as I hear back from the devs.
A simple hack is to apply the patch linked in the issue to the `mplib/planner.py` file after installation.

A coming version of MPLib will include the SD variant, see here: https://github.com/haosulab/MPlib/pull/92
TODO: update call to TOPP-RA in TAPAS.


# Workflow
## Entry points
For a full list of entry points, see `setup.py`.

## Configs
### Structure
Configs are stored as dataclass instances in individual modules in `conf`.
The config structure roughly mirrors the code structure, though that's mostly for ease of use, as subconfigs are imported as modules anyway.
Scripts usually need a config file and some command line args to supplement additional info. (Eg the task is a clarg so that one does not have to create one config file per task.)
Furthermore, any config key can be overwritten from the command line using the `--overwrite` arg. Eg `tapas-eval --config conf/evaluate/franka/banana.py -t Banana -f demos --overwrite wandb_mode=disabled policy.suffix=tx`

### Machine Config
Some config keys are machine specific, such as the data root.
For this, I recommend placing a file called `_machine.py` in the conf folder and import it whenever needed.
Here's an example of how the file may look like.
```
from omegaconf import MISSING

from tapas_gmm.utils.misc import DataNamingConfig

data_naming_config = DataNamingConfig(
    feedback_type=MISSING, task=MISSING, data_root="data"
)


CUDA_PATH = "/usr/local/cuda-11.1"
LD_LIBRARY_PATH = ":".join(
    [CUDA_PATH + "/lib64", "/home/USERNAME/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"]
)
```
Then you can use `from conf._machine import data_naming_config` in any config file to use the correct machine specific data root without need to update your configs across machines.


## Full Example
Install the package as describe above and navigate inside the cloned repository (so that you can easily use the configurations in conf).

```
tapas-collect --config conf/collect_data/franka_banana.py -t Banana

tapas-kp-encode --config conf/kp_encode_trajectories/vit/franka.py --selection_name exp -t Banana -f demos

# run the notebook to fit the model, in this case notebooks/gm/franka/banana.ipynb

tapas-eval--config conf/evaluate/franka/banana.py -t Banana -f demos --overwrite wandb_mode=disabled policy.suffix=tx
```
