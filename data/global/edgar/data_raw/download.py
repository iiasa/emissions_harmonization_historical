from pathlib import Path

import pooch

known_registry = {
    "EDGAR_F-gases_1990_2023.zip": "c58b68a6a8bd8551aa0fd36ed17c4e03c4e1306071483184d6e93cde51cea949",
}

fetcher = pooch.create(
    path=str(Path(__file__).parent),
    base_url="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/EDGAR_2024_GHG/",
    registry=known_registry,
)

for filename in known_registry:
    filename_d = fetcher.fetch(filename, processor=pooch.Unzip())
