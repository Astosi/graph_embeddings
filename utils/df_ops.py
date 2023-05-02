import pandas as pd


def merge_dfs(df1, df2):
    # Merging two dataframes
    frames = [df1, df2]
    merged_df = pd.concat(frames).reset_index(drop=True)

    return merged_df
