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
that are input to the IAM emissions harmonization algorithms in `IAMconsortium/concordia`
(regional harmonization and spatial gridding for ESMs)
and `iiasa/climate-assessment` (global climate emulator workflow).

## Status

- development: the project is actively being worked on

## Installation

We do all our environment management using [pixi](https://pixi.sh/latest).
To get started, you will need to make sure that pixi is installed
([instructions here](https://pixi.sh/latest),
we found that using the pixi provided script was best on a Mac).

To create the virtual environment, run

```sh
pixi install
pixi run pre-commit install
```

These steps are also captured in the `Makefile` so if you want a single
command, you can instead simply run `make virtual-environment`.

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

## The 500x series

Instructions for many of the notebooks are below.
The notebooks that start with `500*`
form a complete set for processing all data.
At present, all the notebooks up until 5029 must be run by hand.

From `5090*` onwards, you can use `scripts/drive-500x-series.py`.
There are 200 lines in this script.
To understand what is going on, you will need to read the script,
there is not extensive documentation.

There is also `scripts/pack-unpack-scenario-running-data.py`.
This can be used to pack up pre-processed data
and then unpack them, so you can skip all the processing
(e.g. biomass burning which takes hours and crashes many laptops).

### Running a new set of scenarios

If you haven't already, login to the scenario explorer with

```sh
pixi run ixmp4 login <your-username>
```

1. In `src/emissions_harmonization_historical/constants_5000.py`,
   increment `DOWNLOAD_SCENARIOS_ID` to a new value.
1. In `scripts/drive-500x-series.py`,
   check that you have specified the notebooks and IAMs you want to run.
1. Run the notebooks with `pixi run python scripts/drive-500x-series.py`
1. Add the downloaded version information with
   `git add -f data/raw/scenarios/<DOWNLOAD_SCENARIOS_ID>/versions.json`

Once things have run, you will probably want to upload results

1. To extract results, there are scripts named
   `scripts/extract*`
1. Pick the scripts you want to run
1. Run the scripts of interest e.g. `pixi run python scripts/extract-climate-results.py`
1. Upload the results to
   [sharepoint](https://iiasahub.sharepoint.com/sites/eceprog/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2Feceprog%2FShared%20Documents%2FProjects%2FCMIP7%2FIAM%20Data%20Processing%2FScenarioMIP%20Emulator%20workflow&viewid=956acd8a%2De1e7%2D4ae9%2Dab1b%2D0506911bae11&csf=1&web=1&e=yQqAUa&CID=2955cf91%2D3d7b%2D4838%2Da9e5%2D37c5c0eae367&FolderCTID=0x012000AA9481BF7BE9264E85B14105F7F082FF)
   in a suitably named folder

## Data

Some of our data is managed using [git lfs](https://git-lfs.com/).
To install it, please follow [the instructions here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

Then, before doing anything else, run

```sh
git lfs install
```

Once you have `git lfs` installed, you can grab all the files we track with

```sh
git lfs fetch --all
```

To grab a specific file, use

```sh
git lfs pull --include="path/to/file"
# e.g.
git lfs pull --include="data/national/gfed/data_aux/iso_mask.nc"
```

For more info, see, for example, [here](https://graphite.dev/guides/how-to-use-git-lfs-pull).

### Input data

Note that this repository focuses on processing data, and does not currently also (re)host input data files.

Files that need to be downloaded to make sure you can run the notebooks are specified in the relevant `data` subfolders,
in README files, such as in `\data\national\ceds\data_raw\README.txt` for the CEDS data download,
and in `\data\national\gfed\data_raw\README.txt` for the GFED data download.

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

### Adding new dependencies

If there is a dependency missing, you can add it with pixi.
Please only add dependencies with pixi,
as this ensures that all the other developers will get the same dependencies as you
(if you add dependencies directly with conda or pip,
then they are not added to the `pixi.lock` file
so other developers will not realise they are needed!).

To add a conda dependency,

```sh
pixi add <dependency-name>
```

To add a PyPI/pip dependency,

```sh
pixi add --pypi <dependency-name>
```

The full documentation can be found [here](https://pixi.sh/v0.24.1/reference/cli/#add)
in case you have a more exotic use case.

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

##### The `500x` series

This series of notebooks is an attempt to clarify the essential bits of the workflow
needed to go from nothing to doing the climate assessment of a submission.
There are lots of other things done in the notebooks,
so we tried to clarify with this series.
The other notebooks are still there and there is duplication with the other notebooks
and some uncertainty about which other notebooks still run.
Sorting this out is a next step.

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

## Original template

This project was generated from this template:
[basic python repository](https://gitlab.com/znicholls/copier-basic-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.
