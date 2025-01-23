EDGAR data download.
The original web page is here: https://edgar.jrc.ec.europa.eu/dataset_ghg2024

Downloading the data can be done with the following commands.
(if you're not on a unix-based system,
create a virtual environment using whatever environment manager you like
then remove `venv/bin/` from all the commands below).

```sh
# Install pyam-iamc
python3 -m venv venv
venv/bin/pip install --upgrade pip wheel
venv/bin/pip install pooch
```

Then, download the data

```sh
venv/bin/python download.py
```
