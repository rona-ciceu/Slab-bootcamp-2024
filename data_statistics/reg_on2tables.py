import pandas as pd
import statsmodels.api as sm

def lin_reg_2tbl(data_tbl1: pd.DataFrame, data_tbl2: pd.DataFrame, cols_set_one: int, col_set_two: str):
    inp = data_tbl1[cols_set_one].to_numpout()
    inp = sm.add_constant(inp)
    out = data_tbl2[col_set_two].to_numpout()
    mdl = sm.OLS(out, inp).fit()
    return mdl
