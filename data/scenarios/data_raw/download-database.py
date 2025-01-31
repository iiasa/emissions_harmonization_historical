import datetime as dt
from pathlib import Path

import pandas as pd
import pyam
import tqdm

OUT_DIR = Path(__file__).parents[0]

time_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

# Establish connection to the legacy pyam platform 'ssp-submission'
pyam.iiasa.Connection("ssp_submission")
conn_ssp = pyam.iiasa.Connection("ssp_submission")
props = conn_ssp.properties().reset_index()

# Retrieve the desired scenarios' emissions
# (download everything; loop over model-scenario combinations to not blow up RAM)
all_meta_l = []
for model, scenario in tqdm.tqdm(
    zip(props["model"], props["scenario"]), total=len(props["model"]), desc="Model-scenario combinations"
):
    print(f"Grabbing {model=} {scenario=}")
    df = pyam.read_iiasa("ssp_submission", model=model, scenario=scenario, variable="Emissions|*", meta=True)
    if df.empty:
        print(f"No data for {model=} {scenario=}")
        continue

    # Write out the scenario data
    output_filename = f"{time_id}__scenarios-scenariomip__{model}__{scenario}.csv".replace("/", "_").replace(" ", "-")
    output_file = OUT_DIR / output_filename
    df.timeseries().to_csv(output_file)
    print(f"Data for {model} {scenario} has been written to {output_file}")

    all_meta_l.append(df.meta)

all_meta = pd.concat(all_meta_l)
all_meta.to_csv(OUT_DIR / f"{time_id}_all-meta.csv")
