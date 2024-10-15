from pathlib import Path

import pooch

known_filenames = [
    *[f"GFED4.1s_{y}.hdf5" for y in range(1997, 2016 + 1)],
    *[f"GFED4.1s_{y}_beta.hdf5" for y in range(2017, 2023 + 1)],
]

fetcher = pooch.create(
    # Download here
    path=str(Path(__file__).parent),
    base_url="https://www.geo.vu.nl/~gwerf/GFED/GFED4/",
    # The registry specifies the files that can be fetched
    registry={
        f: None
        for f in known_filenames
        # "GFED4.1s_1997.hdf5": None,
        # "gravity-disturbance.nc": "sha256:1upodh2ioduhw9celdjhlfvhksgdwikdgcowjhcwoduchowjg8w",
    },
)

for filename in known_filenames:
    fetcher.fetch(filename)
