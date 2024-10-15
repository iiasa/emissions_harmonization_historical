# Processing historical emissions for CMIP7 harmonization routines
<!--- Adding a one-line description of what this repository is for here may be
helpful -->
<!---

We recommend having a status line in your repo to tell anyone who stumbles
on your repository where you're up to. Some suggested options:

- prototype: the project is just starting up and the code is all prototype
- development: the project is actively being worked on
- finished: the project has achieved what it wanted and is no longer being
  worked on, we won't reply to any issues
- dormant: the project is no longer worked on but we might come back to it, if
  you have questions, feel free to raise an issue
- abandoned: this project is no longer worked on and we won't reply to any
  issues

-->

Scripts that combine historical emissions data records from several datasets like CEDS and GFED
to create complete historical emissions files
that are input to the IAM emissions harmonization algorithms in `IAMconsortium/concordia` (regional harmonization and spatial gridding for ESMs)
and `iiasa/climate-assessment` (global climate emulator workflow).

## Status

- prototype: the project is just starting up and the code is all prototype

## Installation

We do all our environment management using [pixi](https://pixi.sh/latest).
To get started, you will need to make sure that pixi is installed
([instructions here](https://pixi.sh/latest),
we found that using the pixi provided script was best on a Mac).

To create the virtual environment, run

```sh
pixi install
pixi run pre-commit install
# Install some specific dependencies which seem unhappy in the rest of the workflow
pixi run pip install "git+https://github.com/gidden/ptolemy.git#egg=ptolemy-iamc" --no-deps
pixi run pip install "git+https://github.com/iiasa/aneris.git@workflow#egg=aneris-iamc" --no-deps
```

These steps are also captured in the `Makefile` so if you want a single
command, you can instead simply run `make virtual-enviroment`.

Having installed your virtual environment, you can now run commands in your
virtual environment using

```sh
pixi run <command>
```

For example, to run Python within the virtual environment, run

```sh
pixi run python
```

As another example, to run a notebook server, run

```sh
pixi run jupyter lab
```

<!--- Other documentation and instructions can then be added here as you go,
perhaps replacing the other instructions above as they may become redundant.
-->

## Data

### Input data
Note that this repository focuses on processing data, and does not currently also (re)host input data files.

Files that need to be downloaded to make sure you can run the notebooks are specified in the relevant `data` subfolders, in README files, such as in `\data\national\ceds\data_raw\README.txt` for the CEDS data download, and in `\data\national\gfed\data_raw\README.txt` for the GFED data download.

### Processed data
Data is processed by the jupyter notebooks (saved as .py scripts using jupytext, under the `notebooks` folder).
The output paths are generally specified at the beginning of each notebook.

For instance, you find processed CEDS data at `\data\national\ceds\processed` and processed GFED data at `\data\national\gfed\processed`.

## Development

<!--- In bigger projects, we would recommend having separate docs where this
development information can go. However, for such a simple repository, having
it all in the README is fine. -->

Install and run instructions are the same as the above (this is a simple
repository, without tests etc. so there are no development-only dependencies).

### Repository structure

#### Notebooks

These are the main processing scripts.
They are saved as plain `.py` files using [jupytext](https://jupytext.readthedocs.io/en/latest/).
Jupytext will let you open the plain `.py` files as Jupyter notebooks.

In general, you should run the notebooks in numerical order.
We do not have a comprehensive way of capturing the dependencies between notebooks implemented at this stage.
We try and make it so that notebooks in each `YY**` series are independent
(i.e. you can run `02**` without running `01**`),
but we do not guarantee this.
Hence, if in doubt, run the notebooks in numerical order.

Overview of notebooks:

- `01**`: preparing input data for `IAMconsortium/concordia`.
- `02**`: preparing input data for `iiasa/climate-assessment`.

#### Local package

We have a local package, `emissions_harmonization_historical`,
that lives in `src`, which we use to share general functions across the notebooks.

#### Data

All data files should be saved in `data`.
We divide data sources into `national` i.e. those that are used for country-level data (e.g. CEDS, GFED)
and `global` i.e. those that are used for global-level data (e.g. GCB).
Within each data source's folder, we use `data_raw` for raw data.
Where raw data is not included, we include a `README.txt` file which explains how to generate the data.

### Tools

In this repository, we use the following tools:

- git for version-control (for more on version control, see
  [general principles: version control](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/theory/version-control.md))
    - for these purposes, git is a great version-control system so we don't
      complicate things any further. For an introduction to Git, see
      [this introduction from Software Carpentry](http://swcarpentry.github.io/git-novice/).
- [Pixi](https://pixi.sh/latest/) for environment management
   (for more on environment management, see
   [general principles: environment management](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/theory/environment-management.md))
    - there are lots of environment management systems.
      Pixi works well in our experience and,
      for projects that need conda,
      it is the only solution we have tried that worked really well.
    - we track the `pixi.lock` file so that the environment
      is completely reproducible on other machines or by other people
      (e.g. if you want a colleague to take a look at what you've done)
- [pre-commit](https://pre-commit.com/) with some very basic settings to get some
  easy wins in terms of maintenance, specifically:
    - code formatting with [ruff](https://docs.astral.sh/ruff/formatter/)
    - basic file checks (removing unneeded whitespace, not committing large
      files etc.)
    - (for more thoughts on the usefulness of pre-commit, see
      [general principles: automation](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/general-principles/automation.md)
    - track your notebooks using
    [jupytext](https://jupytext.readthedocs.io/en/latest/index.html)
    (for more thoughts on the usefulness of Jupytext, see
    [tips and tricks: Jupytext](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/tips-and-tricks/managing-notebooks-jupytext.md))
        - this avoids nasty merge conflicts and incomprehensible diffs
