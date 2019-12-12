 # Manifold MCMC for diffusions
 

Code accompanying the paper [*Manifold MCMC methods for Bayesian inference in a wide class of diffusion models*](https://arxiv.org/abs/1912.02982).

The manifold MCMC methods in the Python package [*Mici*](https://github.com/matt-graham/mici) are used for inference, with the `sde` package in this repository adding some helper functions and classes specific to performing inference in diffusion models.

For a complete example of applying the method described in the paper to perform inference in a Fitzhugh-Nagumo hypoelliptic diffusion model with accompanying explanatory notes see the Jupyter notebook linked below.

<table>
  <tr>
    <th colspan="2"><img src='https://raw.githubusercontent.com/jupyter/design/master/logos/Favicon/favicon.svg?sanitize=true' width="15" style="vertical-align:text-bottom; margin-right: 5px;"/> <a href="FitzHugh-Nagumo_example.ipynb">FitzHugh-Nagumo_example.ipynb</a></th>
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

## Local installation

To install the `sde` package and dependencies to run the notebook locally, first create a local clone of the repository

```bash
git clone https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git
```

Then either create a new Python 3.6+ environment using your environment manager of choice (e.g. [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [`virtualenv`](https://virtualenv.pypa.io/en/latest/userguide/#usage), [`venv`](https://docs.python.org/3/library/venv.html#creating-virtual-environments), [`pipenv`](https://pipenv.kennethreitz.org/en/latest/install/#installing-packages-for-your-project)) or activate the existing environment you wish to use.

To install just the `sde` package and its basic dependencies, from within the `manifold-mcmc-for-diffusions` directory run

```bash
pip install .
```

To install the `sde` package plus all the dependencies required to run the example notebook instead run

```bash
pip install .[notebook]
``` 

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
