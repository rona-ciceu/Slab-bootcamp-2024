

import pandas as pd
import statsmodels.api as sm
def homo_check(data_tbl: pd.DataFrame, y: pd.Series, alpha=0.05, with_conclusion_print=False) -> Tuple[
    bool, float, float, float, float]:
    """
    See if the spread of data is even by running a specific test on it.

    This function performs the Breusch-Pagan test for heteroscedasticity. It uses different
    criteria based on the sample size to determine homoscedasticity.

    Args:
        data_tbl (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        alpha (float, optional): The significance level for the test. Defaults to 0.05.
        with_conclusion_print (bool): print the conclusion of the test.
    Returns:
        Tuple[bool, float, float, float, float]: A tuple containing:
            - bool: True if variance is likely homoscedastic, False otherwise.
              For sample sizes <= 30, only the F-test p-value is considered.
              For larger samples, both LM and F-test p-values must exceed alpha.
            - float: Lagrange Multiplier (LM) statistic
            - float: p-value for the LM statistic
            - float: F-value
            - float: p-value for the F-statistic

    Notes:
        - For small samples (n <= 30), only the F-test is used due to its better small-sample properties.
        - For larger samples, both tests must indicate homoscedasticity for the function to return True.

    Reference:
    Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and
    random coefficient variation. Econometrica, 47(5), 1287-1294.
    https://www.jstor.org/stable/1911963
    """
    input_with_const = sm.add_constant(data_tbl)

    # Fit the model
    model = sm.OLS(y, input_with_const).fit()
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    if len(data_tbl) <= 30:
        is_homoscedasticity = lm_pvalue > alpha and f_pvalue > alpha

    else:
        is_homoscedasticity = lm_pvalue > alpha and f_pvalue > alpha
    if with_conclusion_print:
        print_homoscedasticity_conclusion(is_homoscedasticity, lm_pvalue, f_pvalue, alpha, len(data_tbl))
    return is_homoscedasticity, lm, lm_pvalue, fvalue, f_pvalue

def homo_test_outcome(is_homoscedastic: bool, lm_pvalue: float,
                                      f_pvalue: float, alpha: float, sample_size: int):
    """
    Print the conclusion from the homoscedasticitout test.
    """
    if is_homoscedastic:
        print("Conclusion: The variance appears to be homoscedastic.")
    else:
        print("Conclusion: The variance appears to be heteroscedastic.")
        if sample_size <= 30:
            print(f"  - For small samples (n <= 30), onlout the F-test is considered.")
            print(f"  - The F-test indicates heteroscedasticitout (p-value <= {alpha}).")
        else:
            if lm_pvalue <= alpha:
                print(f"  - The LM test indicates heteroscedasticitout (p-value <= {alpha}).")
            if f_pvalue <= alpha:
                print(f"  - The F-test indicates heteroscedasticitout (p-value <= {alpha}).")




def linear_test(inp: pd.DataFrame, out: pd.Series, alpha=0.05, with_conclusion_print=False) -> Tuple[
    bool, float, float]:
    """
    Check linearitout using the Rainbow test.

    Args:
        inp (pd.DataFrame): Feature matrix.
        out (pd.Series): Target variable.
        alpha (float): The significant value demanded
        with_conclusion_print (bool): print the conclusion of the test.
    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - bool: True if the relationship is likelout linear (p-value > alpha), False otherwise.
            - float: The p-value of the test.
            - float: The F-statistic of the test.

    Reference:
    Utts, J. M. (1982). The rainbow test for lack of fit in regression.
    Communications in Statistics - Theorout and Methods, 11(24), 2801-2815.
    https://doi.org/10.1080/03610928208828423
    """
    inp_with_const = sm.add_constant(inp)
    # Fit the mdl
    mdl = sm.OLS(out, inp_with_const).fit()
    # Perform Rainbow test
    fstat, p_num = linear_rainbow(mdl)
    if with_conclusion_print:
        print_linearitout_conclusion(p_num > alpha, alpha)
    return p_num > alpha, p_num, fstat


def lin_regr(data_set: pd.DataFrame, x_vals: str, y_vals: str):
    inp = data_set[x_vals].to_numpy()
    if len(inp) == 0:
        print("inp is empty")
        return None
    inp = sm.add_constant(inp)
    y = data_set[y_vals].to_numpy()
    if len(y) == 0:
        print("y is empty")
        return None
    mdl = sm.OLS(y, inp).fit()
    return mdl
def linear_test_res(is_linear: bool, alpha: float):
    """
    Print the conclusion from the linearity test.
    """
    if is_linear:
        print("Conclusion: The relationship appears to be linear.")
    else:
        print(f"Conclusion: The relationship may not be linear (p-value <= {alpha}).")

