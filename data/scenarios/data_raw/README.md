Scenario data can be downloaded from IIASA's resources here:
https://data.ece.iiasa.ac.at/ssp-submission.

Downloading the data can be done with the following commands.
(if you're not on a unix-based system,
create a virtual environment using whatever environment manager you like
then remove `venv/bin/` from all the commands below).

```sh
# Install pyam-iamc
python3 -m venv venv
venv/bin/pip install --upgrade pip wheel
venv/bin/pip install pyam-iamc
```

Login with ixmp4

```sh
# Note that this saves your login on your machine in plain text, be careful.
venv/bin/ixmp4 login <your-username>
```

Then, download the data

```sh
venv/bin/python download-database.py
```
