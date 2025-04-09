Biomass burning data is on the ESGF here: https://aims2.llnl.gov/search?project=input4MIPs&activeFacets=%7B%22mip_era%22%3A%22CMIP6Plus%22%2C%22institution_id%22%3A%22DRES%22%7D

Download the data of interest here. (Note that the full dataset is large > 100GB.)
This can be done with the following commands
and [esgpull](https://esgf.github.io/esgf-download/)
(if you're not on a unix-based system,
create a virtual environment using whatever environment manager you like
then remove `venv/bin/` from all the commands below).

```sh
# Install esgpull
# (see note above for non-unix systems)
python3 -m venv venv
venv/bin/pip install esgpull
venv/bin/esgpull self install

# In the diaglog that pops up, install here
# to ensure that the paths in the notebooks behave.
venv/bin/esgpull config api.index_node esgf-node.llnl.gov
```

Then, check that the installation worked correctly by running

```sh
venv/bin/esgpull config
```

The output should look something like

```
─────────────────── /path/to/emissions_harmonization_historical/data/national/gfed-bb4cmip/data_raw/config.toml ───────────────────
[paths]
...
data = "/path/to/emissions_harmonization_historical/data/national/gfed-bb4cmip/data_raw/data"
...

[api]
index_node = "esgf-node.llnl.gov"
...
```

Then, download the data

```sh
# Comma-separated list
# You can search with esgpull search e.g. `venv/bin/esgpull search --all project:input4MIPs mip_era:CMIP6Plus source_id:DRES-CMIP-BB4CMIP7-1-0 grid_label:gn`
variable_ids_to_grab="BC,CH4,CO,CO2,Multiple,N2O,NH3,NMVOC_bulk,NOx,OC,SO2,gridcellarea"
venv/bin/esgpull add --tag bb4cmip --track variable_id:"${variable_ids_to_grab}" project:input4MIPs mip_era:CMIP6Plus source_id:DRES-CMIP-BB4CMIP7-1-0 grid_label:gn
venv/bin/esgpull update -y --tag bb4cmip
venv/bin/esgpull download --tag bb4cmip

venv/bin/esgpull add --tag bb4cmip-nmvoc-bulk --track variable_long_name:"biomass_burning_NMVOC_bulk_flux" project:input4MIPs mip_era:CMIP6Plus source_id:DRES-CMIP-BB4CMIP7-1-0 grid_label:gn
venv/bin/esgpull update -y --tag bb4cmip-nmvoc-bulk
venv/bin/esgpull download --tag bb4cmip-nmvoc-bulk
```

The download takes some time, but if it gets interrupted it can be restarted using
```
venv/bin/esgpull status
```
to check if you have downloads that were not completed (error) or that were not yet started before the interruption (starting)
To put them back in the queue:
```
venv/bin/esgpull retry error
venv/bin/esgpull retry starting
```
and then to continue downloading these:
```
venv/bin/esgpull retry download
```
