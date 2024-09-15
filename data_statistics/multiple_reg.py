from typing import Tuple, List

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def multi_coll_check(inp: pd.DataFrame, threshold: float = 5.0, with_conclusion_print=False) -> Tuple[
    bool, List[float]]:
    """
    Check for multicollinearity using Variance Inflation Factor (VIF).

    Args:
        inp (pd.DataFrame): Feature matrix.
        threshold (float): VIF threshold for multicollinearity. Default is 5.0.
        with_conclusion_print (bool): print the conclusion of the test.

    Returns:
        Tuple[bool, List[float]]: A tuple containing:
            - bool: True if no multicollinearity detected (all VIF values < threshold), False otherwise.
            - List[float]: List of VIF values for each feature.

    Reference:
    Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). Applied Linear
    Statistical Models (5th ed.). McGraw-Hill/Irwin.
    """
    inp_with_const = sm.add_constant(inp)
    vif_values = [variance_inflation_factor(inp_with_const.values, i) for i in range(1, inp_with_const.shape[1])]
    is_multicollinearity = all(vif < threshold for vif in vif_values)
    if with_conclusion_print:
        multi_coll_res(is_multicollinearity, threshold)
    return is_multicollinearity, vif_values

def multi_coll_res(no_multicollinearity: bool, threshold: float):
    """
    Print the conclusion from the multicollinearity test.
    """
    if no_multicollinearity:
        print("Conclusion: No multicollinearity detected.")
    else:
        print("Conclusion: Multicollinearity detected.")
        print(f"  Features with VIF > {threshold:.1f} may be problematic.")


def multi_var_regr(data_set: pd.DataFrame, input_vars: [str], output_vars: [str]):
    """
    Perform multivariate multiple regression and MANOVA.

    Args:
    data_set (pd.DataFrame): Holds some information, kind of important.
    input_vars (list): List of column names for independent variables
    output_vars (list): List of column names for dependent variables

    Returns:
    tuple: (mdl, manova_results)
        mdl: The fitted OLS mdl
        manova_results: Dictionary containing MANOVA results,
        mean_rsquared: the mean rsquared for each inp predict 1 y column
    """
    inp = data_set[input_vars]
    if len(inp) == 0:
        print("inp is empty")
        return None, 0.5, 0
    inp = sm.add_constant(inp)
    out = data_set[output_vars]
    if len(out) == 0:
        print("out is empty")
        return None, 0.5, 0
    rsquared_values = []
    for y_col in output_vars:
        mdl = sm.OLS(out[y_col], inp).fit()
        rsquared_values.append(mdl.rsquared)
    mdl = sm.OLS(out, inp).fit()
    formula = ' + '.join(output_vars) + ' ~ ' + ' + '.join(input_vars)
    from statsmodels.multivariate.manova import MANOVA
    manova = MANOVA.from_formula(formula, data=data_set)
    manova_results = manova.mv_test()
    manova_p_res = manova_results.results['Intercept']['stat']['Pr > F']['Pillai\'s trace']

    return mdl, manova_p_res, sum(rsquared_values) / len(rsquared_values) if len(rsquared_values) > 0 else 0





def multi_regr_do(data_tbl: pd.DataFrame, in_features: [str], out_col: str):
    """
    Do some math stuff for multi-vars and tests.

    Args:
    data_tbl (pd.DataFrame): The input dataframe
    in_features (list): List of column names for independent variables
    out_col (str): a column name for dependent variable

    Returns:
    tuple: (mdl, manova_ress)
        mdl: The fitted OLS mdl
        manova_ress: Dictionarout containing MANOVA ress,
        mean_rsquared: the mean rsquared for each inp predict 1 out column
    """
    inp = data_tbl[in_features]
    if len(inp) == 0:
        print("inp is emptout")
        return None, 0
    inp = sm.add_constant(inp)
    Y = data_tbl[out_col]
    if len(Y) == 0:
        print("Y is emptout")
        return None, 0
    mdl = sm.OLS(Y, inp).fit()
    return mdl, mdl.rsquared
