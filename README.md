<a href="https://colab.research.google.com/drive/1vkp-uPV8tKavmX12bcN2L-jYH8_MgmHL?usp=sharing"><img src="https://img.shields.io/badge/-Colab%20Demo-blue" /></a>
<a href="https://ncsu.qualtrics.com/jfe/form/SV_enuiA15WX8w74qy"><img src="https://img.shields.io/badge/-Caiman%20Survey-green" /></a>

CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">

A Python toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

CaImAn implements a set of essential methods required in the analysis pipeline of large scale calcium imaging data. Fast and scalable algorithms are implemented for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both batch and online modes. CaImAn also contains some routines for the analysis of behavior from video cameras. A list of features as well as relevant references can be found [here](https://caiman.readthedocs.io/en/latest/CaImAn_features_and_references.html).

## Requirements

Right now, CaImAn works and is supported on the following platforms:
* Linux on Intel CPUs
* MacOS on Intel CPUs
* Windows on Intel CPUs

16G RAM is required for a good experience, and depending on datasets, 32G or more may be necessary.

CaImAn presently targets Python 3.8. Parts of CaImAn are written in C++, but apart possibly during install, this is not visible to the user. There is also an [older implementation](https://github.com/flatironinstitute/CaImAn-MATLAB) of CaImAn in Matlab (unsupported). That version can be used with [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 

### Other hardware
* ARM-based versions of Apple hardware work (if on a 16G model), but currently happen under x86 emulation and we cannot support them as well. A native OSX port is planned for late 2021/early 2022.
* Support for Linux on ARM (e.g. AWS Graviton) is not available (but it may work with the port of conda, if you compile Caiman yourself - we do not have binary packages and this is untested). If you care about this, please let us know.


## Install

The supported ways to install CaImAn use the Anaconda python distribution. If you do not already have it, first install a 3.x version for your platform from [here](https://docs.conda.io/en/latest/miniconda.html). Familiarise yourself with Conda before going further.

We strongly recommend installing the mamba package into your base environment, with 'conda install -c conda-forge mamba', using it to build your conda environment. Mamba performs the same environment creation tasks that the base conda tool does, but far faster. In the instructions below, we assume you're using mamba, but if you're not, you can run the same commands with the conda tool instead.

This fork only supports installation from sources (only tested on Linux):

- Clone the sources of this repo
- Create an environment with all the prereqs with 'mamba env create -n caiman -f environment.yml'
- Activate the environment with `conda activate caiman`
- Install pre-requisite modules with 'pip install -e .'

There are other ways to build/use caiman, but they may get less or no support depending on how different they are.

More detailed docs on installation can be found [here](./docs/source/Installation.rst).

After installing the software, the caimanmanager.py script (which will be put in your path on Linux and OSX) is used to unpack datafiles and demos into a directory called caiman\_data. 

If you want to use GPU functionality and have a GPU where you're running CaImAn (most likely a Linux system), you'll want, after you build your conda environment, to switch to a GPU build of the tensorflow package (conda list will tell you, after the version string, what build variant you have - you most likely will get an mlk build, but a "conda search tensorflow" will probably show you some gpu variants you can switch to - pick one appropriate for your conda version, ideally of the same version of tensorflow you otherwise got). If you need help switching versions, reach out to us on the gitter channel.

## Getting Started

If you used caimanmanager to unpack the demos and data files, you will find in the caiman\_data folder a set of demos and jupyter notebooks. demo\_pipeline.py and demo\_behavior.py (or their notebook equivalents) are good introductions to the code.

## Papers and data

### Main paper
A paper explaining most of the implementation details and benchmarking can be found [here](https://elifesciences.org/articles/38173).

```
@article{giovannucci2019caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

All the results and figures of the paper can be regenerated using this package. For more information visit this [page](https://github.com/flatironinstitute/CaImAn/tree/master/use_cases/eLife_scripts).

CaImAn implements a variety of algorithms for analyzing calcium (and voltage) imaging data. A list of references that provide the theoretical background and original code for the included methods can be found [here](https://caiman.readthedocs.io/en/latest/CaImAn_features_and_references.html). 
 
If you use this code please cite the corresponding papers where original methods appeared as well the companion paper.

### Real-time analysis of microendoscopic 1p data

Our online algorithms can be used for real-time analysis of live-streaming data. An example for real-time analysis of microendoscopic 1p data is shown in the notebook `demos/notebooks/demo_realtime_cnmfE.ipynb`.
For more information about the approach check the [paper](https://doi.org/10.1371/journal.pcbi.1008565).

### Analysis pipeline for Voltage Imaging data

VolPy is an analysis pipeline for voltage imaging data. The analysis is based on following objects:

* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
* `VOLPY`: An object for running the spike detection algorithm and saving results.

In order to use VolPy, you must install Keras into your conda environment. You can do this by activating your environment, and then issuing the command "mamba install -c conda-forge keras".

To see examples of how these methods are used, please consult the `demo_pipeline_voltage_imaging.py` script in the `demos/general` folder. For more information about the approach check the [preprint](https://www.biorxiv.org/content/10.1101/2020.01.02.892323v1).

There is also a [general paper](https://journals.plos.org/ploscompbiol/article/comments?id=10.1371/journal.pcbi.1008806) on this pipeline

## Documentation & Wiki

Documentation of the code can be found [here](https://caiman.readthedocs.io/en/master/). 

Other docs:
* [Running CaImAn on a Cluster](docs/CLUSTER.md)
* [Install quirks on some Linux Distributions](docs/README-Distros.md)
* [How CaImAn can use your GPUs](docs/README-GPU.md)

# Main developers:

* Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* Andrea Giovannucci, **University of North Carolina, Chapel Hill**
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Changlia Cai, **University of North Carolina, Chapel Hill**
* Pat Gunn, **Flatiron Institute, Simons Foundation**

A complete list of contributors can be found [here](https://github.com/flatironinstitute/CaImAn/graphs/contributors).

Currently Pat Gunn and Johannes Friedrich are the most active maintainers.


## Questions, comments, issues

For support, you can create a Github issue describing any bugs you wish to report, or any feature requests you may have.

You may also use the [gitter chat room](https://gitter.im/agiovann/Constrained_NMF) for discussion.

Finally, you may reach out via email to one of the primary maintainers (above).

## Acknowledgements

Special thanks to the following people for letting us use their datasets in demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine
* Clay Lacefield, Randy Bruno, Columbia University
* Daniel Aharoni, Peyman Golshani, UCLA

## License

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
