from pathlib import Path

import pooch

# # Used this initially, then switched to a known registry
# known_filenames = [
#     *[f"GFED4.1s_{y}.hdf5" for y in range(1997, 2016 + 1)],
#     *[f"GFED4.1s_{y}_beta.hdf5" for y in range(2017, 2023 + 1)],
# ]
known_registry = {
    "GFED4.1s_1997.hdf5": "997f54a532cae524757c3b35808c10ae0f71ce231c213617cb34ba4b72968bb9",
    "GFED4.1s_1998.hdf5": "36c13cdcec4f4698f3ab9f05bc83d2307252d89b81da5a14efd8e171148a6dc0",
    "GFED4.1s_1999.hdf5": "5d0d18b09d9a76e305522c5b46a97bf3180d9301d1d3c6bfa5a4c838fb0fa452",
    "GFED4.1s_2000.hdf5": "ddbeff2326dded0e2248afd85c3ec7c84a36c6919711632e717d00985cd4ad6d",
    "GFED4.1s_2001.hdf5": "1b684bf0b348e92a5d63ea660564f01439f69c4eb88eacd46280237d51ce5815",
    "GFED4.1s_2002.hdf5": "dcf624961512dbb93759248bc2b75d404b3be68f1f6fdcb01f0c7dc7f11a517a",
    "GFED4.1s_2003.hdf5": "91d61b67d04b4a32d534f5d68ae1de7929f7ea75bb9d25d3273c4d5d75bda4d3",
    "GFED4.1s_2004.hdf5": "931e063f796bf1f7d391d3f03342d2dd2ad1b234cb317f826adfab201003f4cd",
    "GFED4.1s_2005.hdf5": "159e7704d14089496d051546c20b644a443308eeb7d79bf338226af2b4bdc2b7",
    "GFED4.1s_2006.hdf5": "a69d5bf6b8fa3324c2922aac07306ec6e488a850ca4f42d09a397cee30eebd4c",
    "GFED4.1s_2007.hdf5": "1d7f77e6f7b13cc2a8ef9d26ecb9ea3d18e70cfeb8a47e7ecb26f9613888f937",
    "GFED4.1s_2008.hdf5": "bd3771b9b3032d459a79c0da449fdb497cd3400e0e07a0da6b41e930fc5d3e14",
    "GFED4.1s_2009.hdf5": "36ea9b6036cd0ff3672502c3c04180bd209ddb192f86a2e791a2b896308bc5ff",
    "GFED4.1s_2010.hdf5": "5b2d30b5ddc3e20c38c7971faf6791b313b1bbff22e8bc2b14ca7ea9079aa12c",
    "GFED4.1s_2011.hdf5": "fb19c001bef26ca23d07dd8978fd998f4692bdecdec5eb86b91d4b1ffb4a9aa7",
    "GFED4.1s_2012.hdf5": "08033c90295bbc208fac426e01809b68cef62997668085b1e096d8a61ab43e9b",
    "GFED4.1s_2013.hdf5": "cf5249811af4b7099f886e61125dcd15c1127b6125392fe8358d3f0bf8ddb064",
    "GFED4.1s_2014.hdf5": "a293b4c6e03898a0dc184a082a37435673916a02ff02c06668152dcc4d4b8405",
    "GFED4.1s_2015.hdf5": "c043e96a421247afbeb6580fca0bcddf8160180b14d37b13122fc3110534b309",
    "GFED4.1s_2016.hdf5": "2f3b54ff5698ba7f7aa2bb1d4b5e5f95124c0e0db32830ed94aa04bea2cbc2a6",
    "GFED4.1s_2017_beta.hdf5": "a9859da022e97853efd1ce89664f31d1e8c0cddac2f35472d1e445d019f2a927",
    "GFED4.1s_2018_beta.hdf5": "4c08e36c2d7b1bc7b1020a15d36421bdddc68a4cc08ee9f1069d23b49f3cf34b",
    "GFED4.1s_2019_beta.hdf5": "8ac86c5d35e7ddfe9dbe6d71982ea54f6ee3b5a43d9ebbef19ef2957992587e6",
    "GFED4.1s_2020_beta.hdf5": "83b0ba1f5080cd2d19265c05e0c66f9563ad085381541f8c75d31e69ae839b99",
    "GFED4.1s_2021_beta.hdf5": "5bf68b48515b04fe0dbb493c9b2b6e564a5b206da4af499ef5492e4d835516c5",
    "GFED4.1s_2022_beta.hdf5": "46d1e90287c0c012eb1dbb8b7aa2d0c90bfb617b5c708da3c5a8ffa26ca22abb",
    "GFED4.1s_2023_beta.hdf5": "a9c90c311080e3bd2671322369e38d71a478a518777abbea17d2430fb73424ea",
}

fetcher = pooch.create(
    # Download here
    path=str(Path(__file__).parent),
    base_url="https://www.geo.vu.nl/~gwerf/GFED/GFED4/",
    # The registry specifies the files that can be fetched
    # # Used this initially, then switched to a known registry
    # registry={
    #     f: None
    #     for f in known_filenames
    #     # "GFED4.1s_1997.hdf5": None,
    #     # "gravity-disturbance.nc": "sha256:1upodh2ioduhw9celdjhlfvhksgdwikdgcowjhcwoduchowjg8w",
    # },
    registry=known_registry,
)

# # Used this initially, then switched to a known registry
# for filename in known_filenames:
for filename in known_registry:
    filename_d = fetcher.fetch(filename)
    # # Used this initially, then switched to a known registry
    # print(f"'{filename}': '{pooch.file_hash(filename_d)}',")
