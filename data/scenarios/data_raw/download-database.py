import datetime as dt
from pathlib import Path

import pyam

OUT_DIR = Path(__file__).parents[0]

time_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

# Establish connection to the legacy pyam platform 'ssp-submission'
pyam.iiasa.Connection("ssp_submission")
conn_ssp = pyam.iiasa.Connection("ssp_submission")
props = conn_ssp.properties().reset_index()

# Retrieve the desired scenarios' emissions
# (download everything; loop over model-scenario combinations to not blow up RAM)
for model, scenario in zip(props["model"], props["scenario"]):
    print(f"Grabbing {model=} {scenario=}")
    df = pyam.read_iiasa("ssp_submission", model=model, scenario=scenario, variable="Emissions|*")

    # Write out the scenario data
    output_filename = f"{time_id}__scenarios-scenariomip__{model}__{scenario}.csv".replace("/", "_").replace(" ", "-")
    output_file = OUT_DIR / output_filename
    df.to_csv(output_file)
    print(f"Data for {model} {scenario} has been written to {output_file}")
