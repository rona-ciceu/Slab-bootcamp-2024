import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats import t
import tabulate
def desc_stats(data_tbl: pd.DataFrame, columns: [str], with_print=False):
    desc_data_tbl = data_tbl.copy()
    data = [('var', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'ci lower', 'ci upper')]
    for col in columns:
        col_data = desc_data_tbl[~desc_data_tbl[col].isna()][col].astype(int)
        desc = col_data.describe()
        sums = col_data.tolist()
        mean = np.mean(sums)
        s = sem(sums)
        ci = t.interval(0.95, len(sums) - 1, loc=mean, scale=s)
        data.append((col, desc['count'], mean, np.std(sums), desc['min'], desc['25%'], desc['50%'], desc['75%'],
                     desc['max'], ci[0], ci[1]))

    if with_print:
        print(tabulate(data[1:], headers=data[0], tablefmt='fancy_grid'))
    return data
