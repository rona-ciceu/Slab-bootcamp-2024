def anova_fdr(data_set: pd.DataFrame, cat_var: str, cont_var: str):
    """
    Do some stats on two different types of data and fix the results.

    Parameters:
    data (pd.DataFrame): The input data frame.
    cat_var (str): The name of the categorical variable.
    cont_var (str): The name of the continuous variable.

    Returns:
    tuple: ANOVA test results, pairwise comparison results, FDR adjusted p-values.
    """
    # Define the mdl formula
    formula = f'{cont_var} ~ C({cat_var})'

    # Fit the mdl
    mdl = ols(formula, data=data_set).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(mdl, typ=2)

    # Perform pairwise comparisons using Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=data_set[cont_var], groups=data_set[cat_var], alpha=0.05)

    # Extract p-values from the pairwise comparison results
    p_ress = tukey_result.pvalues

    # Perform FDR adjustment on the p-values
    _, pvals_corrected, _, _ = multipletests(p_ress, alpha=0.05, method='fdr_bh')
    tukey_result_data = np.array(tukey_result._results_table.data)
    # Create a DataFrame for the pairwise comparison results with FDR adjusted p-values
    pairwise_results = pd.DataFrame(data={
        'group1': tukey_result_data[1:, 0],
        'group2': tukey_result_data[1:, 1],
        'meandiff': tukey_result_data[1:, 2],
        'p-adj': tukey_result_data[1:, 3],
        'lower': tukey_result_data[1:, 4],
        'upper': tukey_result_data[1:, 5],
        'reject': tukey_result_data[1:, 6],
        'pvals_corrected': pvals_corrected
    })

    return anova_table, pairwise_results









def test_feat_combos(data_set, feature_columns, target_column, num_features=2, with_conclusion_print=False):
    for feature_combo in combinations(feature_columns, num_features):
        inp = data_set[list(feature_combo)]
        y = data_set[target_column]
        mdl = sm.OLS(y, sm.add_constant(inp.to_numpy())).fit()
        residuals = np.array(mdl.resid)
        y_pred = mdl.predict(sm.add_constant(inp.to_numpy()))
        is_normal, JB, p_res, skewness, kurtosis = check_normality(residuals,
                                                                     with_conclusion_print=with_conclusion_print)
        plot_normality_test(residuals,feature_combo,target_column ,is_normal, JB, p_res, skewness, kurtosis)
        is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue = check_homoscedasticity(y, y_pred,
                                                                                   with_conclusion_print=with_conclusion_print)
        homo_test_plot(y, y_pred,feature_combo,target_column, is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue)

        is_linear, lin_p_res, fstat = check_linearity(inp, y, with_conclusion_print=with_conclusion_print)
        plot_linearity_test(inp, y.values,feature_combo, target_column, is_linear, lin_p_res, fstat)

        no_multicollinearity, vif_values = multi_coll_check(inp, with_conclusion_print=with_conclusion_print)
        plot_multicollinearity_test(vif_values, threshold=5.0)

        no_autocorrelation, lb_p_res, dw_statistic = check_autocorrelation(residuals,
                                                                             with_conclusion_print=with_conclusion_print)
        auto_corr_plot(residuals,feature_combo,target_column, no_autocorrelation, lb_p_res, dw_statistic)









