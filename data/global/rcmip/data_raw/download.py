from pathlib import Path

import pooch

known_registry = {
    "rcmip-emissions-annual-means-v5-1-0.csv": "2af9f90c42f9baa813199a902cdd83513fff157a0f96e1d1e6c48b58ffb8b0c1",
}

fetcher = pooch.create(
    path=str(Path(__file__).parent),
    base_url="https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/",
    registry=known_registry,
)

for filename in known_registry:
    filename_d = fetcher.fetch(filename)
