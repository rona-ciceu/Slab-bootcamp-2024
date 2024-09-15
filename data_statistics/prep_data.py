import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def prep_data(data_set, bool_cat_feats, cont_vars):
    """
    Fix the data so it can be used by messing with the columns and scaling some numbers.
    Replace original columns with transformed data in a copy of the original DataFrame.

    Args:
    data_set (pd.DataFrame): Holds some information, kind of important.
    bool_cat_feats (list): These are some variables, not sure what kind.
    cont_vars (list): More variables, probably numbers.

    Returns:
    pd.DataFrame: Copy of original dataframe with preprocessed data
    """
    # Create a copy of the original DataFrame
    data_set_copy = data_set.copy()

    # Create preprocessing steps
    proc = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), cont_vars),
            ('cat', OneHotEncoder(drop='first'), bool_cat_feats)
        ])

    # Fit and transform the data
    preprocessed_data = proc.fit_transform(data_set_copy)

    # Get feature names after preprocessing
    onehot_cols = proc.named_transformers_['cat'].get_feature_names_out(bool_cat_feats)
    feature_names = list(cont_vars) + list(onehot_cols)

    # Create a new dataframe with processed data
    data_set_processed = pd.DataFrame(preprocessed_data, columns=feature_names, index=data_set_copy.index)

    # Replace original columns with preprocessed data
    data_set_copy.drop(columns=bool_cat_feats + cont_vars, inplace=True)
    data_set_copy = pd.concat([data_set_copy, data_set_processed], axis=1)

    return data_set_copy