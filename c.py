def manova_eta_fdr(data_tbl: pd.DataFrame, dep_vars: [str], cat_vars: [str]):
    """
    Do a multi-test, adjust stuff, and get some squared number at the end.

    Parameters:
    data (pd.DataFrame): The input data frame.
    dep_vars (list of str): The names of the dependent variables.
    cat_vars (list of str): The names of the categorical variables.

    Returns:
    tuple: MANOVA results, ANOVA test results with eta squared, pairwise comparison results with FDR adjusted p-values.
    """
    categorical_frm = ' + '.join(cat_vars)
    frm = f'{"+".join(dep_vars)} ~ {categorical_frm}'
    manova = MANOVA.from_frm(frm, data=data_tbl)

    stat_res = {}
    pairwise_results = {}
    eta_num_results = {}

    for dependent_var in dep_vars:
        # Define the ANOVA frm for each dependent variable
        anova_frm = f'{dependent_var} ~ {categorical_frm}'

        # Fit the ANOVA model
        model = ols(anova_frm, data=data_tbl).fit()

        # Perform ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Calculate eta squared
        ss_effect = anova_table['sum_sq'].iloc[0]
        ss_total = anova_table['sum_sq'].sum()
        eta_num = ss_effect / ss_total

        # Add eta squared to the ANOVA table
        anova_table['eta_sq'] = [eta_num] + [None] * (len(anova_table) - 1)

        # Store ANOVA table
        stat_res[dependent_var] = anova_table

        # Perform pairwise comparisons using Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(endog=data_tbl[dependent_var], groups=data_tbl[cat_vars[0]], alpha=0.05)

        # Extract p-values from the pairwise comparison results
        p_values = tukey_result.pvalues

        # Perform FDR adjustment on the p-values
        _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        tukey_result_data = np.array(tukey_result._results_table.data)
        # Create a DataFrame for the pairwise comparison results with FDR adjusted p-values
        pairwise_result = pd.DataFrame(data={
            'group1': tukey_result_data[1:, 0],
            'group2': tukey_result_data[1:, 1],
            'meandiff': tukey_result_data[1:, 2],
            'p-adj': tukey_result_data[1:, 3],
            'lower': tukey_result_data[1:, 4],
            'upper': tukey_result_data[1:, 5],
            'reject': tukey_result_data[1:, 6],
            'pvals_corrected': pvals_corrected
        })

        # Store pairwise results
        pairwise_results[dependent_var] = pairwise_result

    return manova, stat_res, pairwise_results


def lasso_regr(data_tbl: pd.DataFrame, in_features: [str], out_features: [str]):
    input = data_tbl[in_features]
    output = data_tbl[out_features]
    model = MultiTaskLassoCV().fit(input, output)
    return model




def auto_corr_check(leftovers: np.ndarray, alpha=0.05, with_conclusion_print=False) -> Tuple[bool, float, float]:
    """
    Check for autocorrelation using the Durbin-Watson test and Ljung-Box test.

    Args:
        leftovers (np.ndarray): The leftovers from a linear regression model.
        alpha (float): The significance level for the test.
        with_conclusion_print (bool): print the conclusion of the test.

    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - bool: True if no significant autocorrelation (Ljung-Box p-value > alpha), False otherwise.
            - float: The p-value of the Ljung-Box test.
            - float: The Durbin-Watson statistic.

    References:
    Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least
    squares regression. II. Biometrika, 38(1/2), 159-177.
    https://www.jstor.org/stable/2332325

    Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models.
    Biometrika, 65(2), 297-303.
    https://doi.org/10.1093/biomet/65.2.297
    :param with_conclusion_print:
    """
    dw_statistic = durbin_watson(leftovers)
    lb_test = acorr_ljungbox(leftovers, lags=[1], return_data_tbl=True)
    lb_p_value = lb_test['lb_pvalue'].values[0]
    if with_conclusion_print:
        print_autocorrelation_conclusion(lb_p_value > alpha, lb_p_value, dw_statistic, alpha)
    return lb_p_value > alpha, lb_p_value, dw_statistic




def cat_count_plot(data_tbl, category_column, subplot_column=None, title="", x_label=""):
    """
    Counts the number of items in each category of the specified column in the DataFrame
    and plots the counts as a bar chart. If subplot_column is provided, creates a subplot
    for each unique value in the subplot_column.

    Parameters:
    data_tbl (pd.DataFrame): The DataFrame containing the data.
    category_column (str): The name of the category column to count items in.
    subplot_column (str, optional): The name of the column to create subplots for. Default is None.

    Returns:
    dict: A dictionary with the counts of each category for each subplot (if subplot_column is provided).
    """
    plot_data_tbl = data_tbl.copy()
    if category_column not in plot_data_tbl.columns:
        raise ValueError(f"Column '{category_column}' does not exist in the DataFrame.")

    if subplot_column and subplot_column not in plot_data_tbl.columns:
        raise ValueError(f"Column '{subplot_column}' does not exist in the DataFrame.")
    palette = palettes[category_column] if category_column in palettes else None
    order = orders[category_column] if category_column in orders else None

    if subplot_column:
        unique_values = plot_data_tbl[subplot_column].unique()
        num_subplots = len(unique_values)
        fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6))
        counts_dict = {}

        for i, value in enumerate(unique_values):
            ax = axes[i] if num_subplots > 1 else axes
            ax.set_facecolor("white")
            subset = plot_data_tbl[plot_data_tbl[subplot_column] == value]
            counts = subset[category_column].value_counts()
            if order:
                index_order = sorted(counts.index.to_list(), key=lambda x: order[x])
                counts = counts.reindex(index_order)
            colors = ['#FF5733'] * len(counts)
            if palette is not None:
                for j, col in enumerate(counts.to_dict().keys()):
                    colors[j] = palette[col]
            total = counts.sum()
            counts.plot(kind='bar', ax=ax, color=colors)
            for p in ax.patches:
                height = p.get_height()
                percentage = 100 * height / total
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height,
                    f'{percentage:.1f}%\nN:{height}',
                    ha='center',
                    va='bottom'
                )
            ax.set_suptitle(title)
            if x_label:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(category_column)
            ax.set_ylabel('')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center')
            ax.set_yticks([])
            counts_dict[value] = counts
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show()
        return counts_dict

    else:
        category_counts = plot_data_tbl[category_column].value_counts()
        if order:
            index_order = sorted(category_counts.index.to_list(), key=lambda x: order[x])
            category_counts = category_counts.reindex(index_order)
        plt.figure(figsize=(10, 6))
        colors = ['#FF5733'] * len(category_counts)
        if palette is not None:
            for i, col in enumerate(category_counts.to_dict().keys()):
                colors[i] = palette[col]
        category_counts.plot(kind='bar', color=colors)
        total = category_counts.sum()
        ax = plt.gca()
        ax.set_facecolor("white")
        for p in plt.gca().patches:
            height = p.get_height()
            percentage = 100 * height / total
            ax.text(
                p.get_x() + p.get_width() / 2,
                height,
                f'{percentage:.1f}%\nN:{height}',
                ha='center',
                va='bottom'
            )
        plt.suptitle(title)
        if x_label:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel(category_column)
        plt.ylabel('')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(rotation=30, ha='center')
        plt.yticks([])
        plt.show()

        return category_counts


def interact_plot(data_tbl: pd.DataFrame, first_cat: str, second_cat: str, depended_vars: list,
                                       aggregation_function):
    # Get the unique levels of the categorical variables
    plot_data_tbl = data_tbl.copy()
    if data_tbl[first_cat].nunique() < data_tbl[second_cat].nunique():
        first_cat, second_cat = second_cat, first_cat
    for cat in [first_cat, second_cat]:
        if len(plot_data_tbl[cat].unique()) > 2:
            continue
        if cat == 'gender':
            plot_data_tbl[cat] = plot_data_tbl[cat].map({0: 'female', 1: 'male'})
        else:
            plot_data_tbl[cat] = plot_data_tbl[cat].astype(bool)
    first_cat_levels = plot_data_tbl[first_cat].unique()
    second_cat_levels = plot_data_tbl[second_cat].unique()

    # Generate markers and colors dynamically based on the unique levels
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'x', 'input', '+',
               'd']  # A list of markers to choose from
    colors = plt.cm.rainbow(np.linspace(0, 1, len(second_cat_levels)))  # Generate a list of colors

    # Create subplots
    fig, axes = plt.subplots(1, len(depended_vars), figsize=(15, 5))

    for i, depended_var in enumerate(depended_vars):
        interaction_plot(
            plot_data_tbl[first_cat].to_numpy(),
            plot_data_tbl[second_cat].to_numpy(),
            plot_data_tbl[depended_var].to_numpy(),
            xlabel=first_cat,
            ylabel=second_cat,
            markers=[markers[j % len(markers)] for j in range(len(second_cat_levels))],  # Ensure we have enough markers
            colors=colors,  # Use the dynamically generated colors
            ms=10,
            ax=axes[i],
            func=aggregation_function
        )
        axes[i].set_title(f'Interaction Plot for {depended_var}')

    plt.tight_layout()
    plt.show()


palettes = {
    'type': {'E': '#BDD997', 'C': '#984788'},
    'substances_main_group': {
        "No Use": "#5765ab", "Light": "#50C878", "MDMA": "#F43183", "Stim": "#FF7F50", "Haluc": "#9966CC",
        "Other": '#DAA520'
    },
    'questionnaire': {'followup 1': 'lightgray', 'followup 2': 'lightgray'},
    'Got back to work since TE': {
        'didnt answered': '#e6194B',  # Bright Red
        'yes': '#3cb44b',  # Moderate Green
        'partially': '#ffe119',  # Bright outputellow
        'no': '#4363d8',  # Strong Blue
        'didnt worked before the event': '#f58231',  # Vivid Orange
        'yes, in the 3 month after TE': '#911eb4',  # Dark Violet
        'yes, in the last month': '#42d4f4',  # Sky Blue for yes in the last month
        'other': '#fabed4'  # Light Pink for other
    },
    'Got back to work since TE summarized': {
        'didnt answered': '#ff6347',  # Tomato Red
        'yes': '#2E8B57',  # Sea Green
        'partially': '#ffeb3b',  # Bright outputellow
        'no': '#DC143C'  # Crimson
    },
    'state change': {
        'yes -> yes': '#2E8B57',
        'no -> yes': '#2E8B57',
        'no -> no': '#DC143C',
        'yes -> no': '#DC143C',
    }
}
orders = {
    'Got back to work since TE summarized': {
        'didnt answered': 3,
        'yes': 0,
        'partially': 1,
        'no': 2
    },
    'substances_main_group': {
        "No Use": 0, "Light": 1, "MDMA": 4, "Stim": 3, "Haluc": 5,
        "Other": 6
    },
    'state change': {
        'yes -> yes': 0,
        'no -> yes': 1,
        'no -> no': 3,
        'yes -> no': 2,
    }
}


def hist_kde_plot(data_tbl: pd.DataFrame, column: str, bins: int = 10, title: str = None, xlabel: str = None,
                            ylabel: str = 'Frequency') -> list:
    """
    Plot a histogram with a Kernel Density Estimate (KDE) for a specified column in a pandas DataFrame.

    Parameters:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to plot the histogram with KDE for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    a list of the bins of the plot

    Raises:
    KeyError: If the specified column does not exist in the DataFrame.
    TypeError: If the input DataFrame is not a pandas DataFrame.

    Example:
    >>> data_tbl = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 4, 5]})
    >>> hist_kde_plot(data_tbl, 'values', bins=5, title='Histogram with KDE', xlabel='Values')
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if column not in data_tbl.columns:
        raise KeyError(f"The column '{column}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data_tbl[column].dropna(), bins=bins, kde=True)
    if title:
        plt.title(title)
    else:
        plt.title(f'histogram with kde for {column},N:{len(data_tbl[column].dropna())}')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    bar_patches = ax.patches
    bin_edges = [patch.get_x() for patch in bar_patches]
    return bin_edges


def group_hist_plot(data_tbl: pd.DataFrame, columns: list, bins: int = 10, title: str = None, xlabel: str = None,
                           ylabel: str = 'Frequency') -> np.array:
    """
    Plot grouped histograms for specified columns in a pandas DataFrame.

    Parameters:
    --------
    - data_tbl (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): The list of columns to plot grouped histograms for.
    - bins (int, optional): Number of bins for the histogram. Default is 10.
    - title (str, optional): The title of the histogram. Default is None.
    - xlabel (str, optional): The label for the x-axis. Default is None.
    - ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    --------
    the bins of the histogram
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    for column in columns:
        if column not in data_tbl.columns:
            raise KeyError(f"The column '{column}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data_tbl[columns].dropna(), bins=bins, alpha=0.5, edgecolor='black', label=columns,
                                stacked=True)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    return bins


def lasso_multi_plot(data_tbl: pd.DataFrame, in_features: [str], out_features: [str]):
    input = data_tbl[in_features]
    output = data_tbl[out_features]
    model = MultiTaskLassoCV().fit(input, output)
    output_pred = model.predict(input)
    data = pd.DataFrame(input, columns=in_features)
    data[out_features] = output
    data[[f'{col}_pred' for col in out_features]] = output_pred

    sns.pairplot(data)
    plt.suptitle('Pairplot of Predictors and Responses')
    plt.show()


def linear_plot(data_tbl: pd.DataFrame, y: np.array, feature_combo: [str], target_column: str, is_linear: bool,
                        p_value: float, fstat: float):
    """
    Plot the results of the linearity test.
    """
    n_features = data_tbl.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

    if n_features == 1:
        axes = [axes]  # Make axes iterable when there's only one feature
    for i, column in enumerate(data_tbl.columns):
        axes[i].scatter(data_tbl[column], y)
        axes[i].set_title(f'{column} vs {target_column}')
        axes[i].set_xlabel(column)
        if i == 0:
            axes[i].set_ylabel(f'{target_column}')
    plt.suptitle(
        f'Linearity Test Results (F={fstat:.2f}, p={p_value:.4f}) are: {"linear" if is_linear else "not linear"}')
    plt.title(f'[{",".join(feature_combo)}] vs {target_column}')
    plt.tight_layout()
    plt.show()
