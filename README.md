 # Manifold MCMC for diffusions

Code accompanying the paper [*Manifold Markov chain Monte Carlo methods for Bayesian inference in a wide class of diffusion models*](https://arxiv.org/abs/1912.02982).

The manifold Markov chain Monte Carlo (MCMC) methods in [Mici](https://github.com/matt-graham/mici) are used for inference, with the `sde` package in this repository adding helper functions and classes specific to performing inference in stochastic differential equation (SDE) models, in particular diffusion processes. The numerical integrators for SDE systems in `sde.integrators` and example models in `sde.example_models` use [SymNum](https://github.com/matt-graham/symnum) to automatically construct derivatives of model specific functions. [JAX](https://github.com/google/jax) is used to generate efficient just-in-time compiled functions for numerically integrating the diffusion models and evaluating the derivatives required for performing MCMC inference.

## Installation

The `sde` package requires Python 3.7 or above. To install the `sde` package and its dependencies in the current Python environment run

```bash
pip install git+https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git
```

To install the package and all dependencies required to run the [scripts for reproducing the experiments in the paper](#experiment-scripts) run

```bash
pip install git+https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git#egg=sde\[scripts\]
```

To install the package and all dependencies required to run [the example Jupyter notebook](#example-notebook) run


```bash
pip install git+https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git#egg=sde\[notebook\]
```


## Experiment scripts

A number of scripts for reproducing the numerical experiments used to produce the figures in the paper are provided in the `scripts` directory. To run these scripts the `sde` package, its dependencies and the additional dependencies required to run the scripts need to be installed in the local Python environment [as described above](#installation).

For the experiments with the FitzHugh&ndash;Nagumo model with noisy observations, the MCMC algorithm implemented in the Julia [BridgeSDEInference](https://github.com/mmider/BridgeSDEInference.jl) package is also used for comparison. A local Julia installation (tested with version 1.6) is therefore also required to run these experiments. The required Julia dependencies can be installed from a Julia REPL by running

```Julia
import Pkg
Pkg.add(["ArgParse", "BridgeSDEInference", "JSON", "NPZ", "PyCall"])
```

There are Python scripts for running individual experiments with each model and inference algorithm combination, as well as Python scripts for generating plots from the results for each model. A description of what a particular script does, what arguments can be passed and what their default values are can be displayed by running

```bash
python path/to/script.py --help
```

There are also bash scripts provided for running the full set of experiments used to produce the figures in the paper. As a warning, these scripts sequentially run experiments over grids of different variables and in some cases for multiple different pseudo-random seeds, and so running all these scripts should be expected to take several days to complete. As some of the plotted values are dependent on computation times which is affected among other things by the system being run on, the generated plots will not exactly match those in the paper.


## Example notebook

A complete example of applying the method described in the paper to perform inference in a Fitzhugh-Nagumo hypoelliptic diffusion model with accompanying explanatory notes is provided in the Jupyter notebook [`FitzHugh-Nagumo_example.ipynb`](FitzHugh-Nagumo_example.ipynb) . This notebook can also be viewed or run interactively online using the links below.

<table>
  <tr>
    <th colspan="2"><img src='https://raw.githubusercontent.com/jupyter/design/master/logos/Favicon/favicon.svg?sanitize=true' width="15" style="vertical-align:text-bottom; margin-right: 5px;"/> FitzHugh-Nagumo_example.ipynb</th>
  </tr>
  <tr>
    <td>Open non-interactive version with nbviewer</td>
    <td>
      <a href="https://nbviewer.jupyter.org/github/thiery-lab/manifold-mcmc-for-diffusions/blob/master/FitzHugh-Nagumo_example.ipynb">
        <img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg?sanitize=true" width="109" alt="Render with nbviewer"  style="vertical-align:text-bottom" />
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Binder</td>
    <td>
      <a href="https://mybinder.org/v2/gh/thiery-lab/manifold-mcmc-for-diffusions/master?filepath=FitzHugh-Nagumo_example.ipynb">
        <img src="https://mybinder.org/badge_logo.svg" alt="Launch with Binder"  style="vertical-align:text-bottom"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Google Colab</td>
    <td>
      <a href="https://colab.research.google.com/github/thiery-lab/manifold-mcmc-for-diffusions/blob/master/FitzHugh-Nagumo_example.ipynb">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
       </a>
    </td>
  </tr>
</table>

## Citation

To cite the pre-print the following `bibtex` entry can be used

```bibtex
@article{graham2019manifold,
  author={Graham, Matthew M. and Thiery, Alexandre H. and Beskos, Alexandros},
  title={Manifold Markov chain Monte Carlo methods for Bayesian inference in a wide class of diffusion models},
  year={2019},
  journal={Pre-print arxiv:1912.02982},
  url={https://arxiv.org/abs/1912.02982}
}
```
