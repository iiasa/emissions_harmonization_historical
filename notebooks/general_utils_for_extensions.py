import numpy as np
import pandas as pd
import pandas_indexing as pix


def interpolate_to_annual(idf: pd.DataFrame, max_supplement: float = 1e-5) -> pd.DataFrame:
    """
    Interpolate pd.DataFrame of emissions input that might not have data for all years
    """
    # TODO: push into pandas-openscm
    missing_cols = np.setdiff1d(np.arange(idf.columns.min(), idf.columns.max() + max_supplement), idf.columns)

    out = idf.copy()
    out.loc[:, missing_cols] = np.nan
    out = out.sort_index(axis="columns").T.interpolate("index").T

    return out


def glue_with_historical(scen_df: pd.DataFrame, hist_df: pd.DataFrame, history_end=2023) -> pd.DataFrame:
    """
    Glue historical data to the beginning of scenario data to get complete timeseries
    """
    out = interpolate_to_annual(scen_df.copy())
    orig_len = len(out.columns)
    missing_years = np.arange(int(hist_df.columns[0]), int(scen_df.columns[0]))
    out.loc[:, missing_years] = np.nan  # hist_df.loc[:, missing_years].values
    for index, df_row in out.iterrows():
        df_row.iloc[orig_len:] = hist_df.loc[pix.ismatch(variable=index[3])].values[
            0, : int(scen_df.columns[0] - hist_df.columns[0])
        ]
        for year in range(scen_df.columns[0], history_end + 1):
            if np.isnan(df_row.iloc[year - scen_df.columns[0]]):
                df_row.iloc[year - scen_df.columns[0]] = hist_df.loc[pix.ismatch(variable=index[3])].values[
                    0, int(year - hist_df.columns[0])
                ]
        out.loc[index, :] = df_row.values
    out = out.sort_index(axis="columns")
    return out
