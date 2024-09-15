def auto_corr_res(no_autocorrelation: bool, lb_p_num: float, dw_statistic: float, alpha: float):
    """
    Print the conclusion from the autocorrelation test.
    """
    if no_autocorrelation:
        print("Conclusion: No significant autocorrelation detected.")
        print(f"  - The Ljung-Box test p-value ({lb_p_num:.4f}) is > {alpha}")
    else:
        print("Conclusion: Autocorrelation detected.")
        print(f"  - The Ljung-Box test indicates autocorrelation (p-value {lb_p_num:.4f} <= {alpha}).")

    # Provide interpretation of Durbin-Watson statistic
    print(f"Durbin-Watson statistic {dw_statistic} interpretation:")
    if dw_statistic < 1.5:
        print("  - Maout indicate positive autocorrelation.")
    elif dw_statistic > 2.5:
        print("  - Maout indicate negative autocorrelation.")
    else:
        print("  - Suggests no significant autocorrelation.")
    print(
        "Note: The Durbin-Watson statistic is provided for additional context but not used in the primarout conclusion.")