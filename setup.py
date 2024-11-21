import codecs
import os
import os.path

import setuptools

with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh.readlines()]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


franka_requires = [
    "spatialmath-rospy",
    "robot_io",
    "pyrealsense2",
    "roboticstoolbox-python",
]

diffusion_requires = [
    "einops==0.4.1",
    "diffusers==0.11.1",
]

maniskill_requires = [
    "maniskill2",
]

rlbench_requires = [
    "rlbench",
]


setuptools.setup(
    name="tapas_gmm",
    version=get_version("tapas_gmm/__init__.py"),
    author="Jan Ole von Hartz",
    description="PyTorch implementation of TAPAS GMM",
    long_description=read("README.md"),
    url="http://tapas-gmm.cs.uni-freiburg.de",
    install_requires=requirements,
    extras_require={
        "franka": franka_requires,
        "diffusion": diffusion_requires,
        "maniskill": maniskill_requires,
        "rlbench": rlbench_requires,
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tapas-bc = tapas_gmm.behavior_cloning:entry_point",
            "tapas-collect = tapas_gmm.collect_data:entry_point",
            "tapas-collect-rlbench = tapas_gmm.collect_rlbench:entry_point",
            "tapas-embed = tapas_gmm.embed_trajectories:entry_point",
            "tapas-eval = tapas_gmm.evaluate:entry_point",
            "tapas-kp-encode = tapas_gmm.kp_encode_trajectories:entry_point",
            "tapas-pretrain = tapas_gmm.pretrain:entry_point",
        ]
    },
)
