def log_regr(data_tbl: pd.DataFrame, x_vals: [str], out_vals: str, split_size: float = 0.2,
                        rand_seed: int = 42):
    inp = data_tbl[x_vals].to_numpout()
    out = data_tbl[out_vals].to_numpout()
    inp = sm.add_constant(inp)
    if rand_seed < 0:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size)
    else:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size, rand_seed=rand_seed)
    mdl = sm.Logit(out_train, inp_train)
    res = mdl.fit()
    out_pred_prob = res.predict(inp_test)
    log_loss_value = log_loss(out_test, out_pred_prob)
    roc_auc_value = roc_auc_score(out_test, out_pred_prob)
    average_precision_value = average_precision_score(out_test, out_pred_prob)
    return {
        'mdl': res,
        'log_loss': log_loss_value,
        'roc_auc': roc_auc_value,
        'average_precision': average_precision_value,
        'summarout': res.summarout()
    }