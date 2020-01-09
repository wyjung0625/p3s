# Population-Guided Parallel Policy Search (P3S)
The algorithm is based on the paper "Population-Guided Parallel Policy Search for Reinforcement Learning" submitted to ICLR 2020.
The P3S codes are modified from the code of Soft Actor-Critic (SAC) (https://github.com/haarnoja/sac)

# Getting Started

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllab
```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. [Download](https://www.roboti.us/index.html) and copy mujoco files to rllab path:
If you're running on OSX, download https://www.roboti.us/download/mjpro131_osx.zip instead, and copy the `.dylib` files instead of `.so` files.
```
mkdir -p /tmp/mujoco_tmp && cd /tmp/mujoco_tmp
wget -P . https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mkdir <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libmujoco131.so <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libglfw.so.3 <installation_path_of_your_choice>/rllab/vendor/mujoco
cd ..
rm -rf /tmp/mujoco_tmp
```

3. Copy your Mujoco license key (mjkey.txt) to rllab path:
```
cp <mujoco_key_folder>/mjkey.txt <installation_path_of_your_choice>/rllab/vendor/mujoco
```

4. Go to "p3s_iclr2020" directory
```
cd <p3s_iclr2020_folder>
```

5. Create and activate conda environment
```
cd p3s_iclr2020 # TODO.before_release: update folder name
conda env create -f environment.yml
source activate p3s
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
source deactivate
conda remove --name p3s --all
```

## Examples
### Training and simulating an agent
```
python ./examples/mujoco_all_p3s_td3.py --env=ant
python ./examples/mujoco_all_p3s_td3.py --env=half-cheetah
python ./examples/mujoco_all_p3s_td3.py --env=hopper
python ./examples/mujoco_all_p3s_td3.py --env=walker
python ./examples/mujoco_all_p3s_td3.py --env=delayed_ant
python ./examples/mujoco_all_p3s_td3.py --env=delayed_half-cheetah
python ./examples/mujoco_all_p3s_td3.py --env=delayed_hopper
python ./examples/mujoco_all_p3s_td3.py --env=delayed_walker
```


