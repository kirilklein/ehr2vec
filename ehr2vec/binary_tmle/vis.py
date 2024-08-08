def display_results(models, diffs, stds):    
    # come up with a nice scheme which has at max 3 subplots per row, based on the number of models
    n_rows = len(models) // 3 + 1 if len(models) % 3 != 0 else 0
    n_cols = 3 if len(models) >= 3 else len(models)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    estimators = list(diffs.keys())
    ax = ax.flatten()
    for i, (model_name, model) in enumerate(models.items()):
        ax[i].text(0.35, 0.1, f"Alpha: {model['alpha']}\nBeta: {model['beta']}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax[i].transAxes, fontsize=12)
        for estimator in estimators:
            ax[i].plot(patient_numbers, diffs[estimator][model_name], label=estimator)

            upper_boundary = diffs[estimator][model_name] + stds[estimator][model_name]
            lower_boundary = diffs[estimator][model_name] - stds[estimator][model_name]
            ax[i].fill_between(patient_numbers, upper_boundary, lower_boundary, alpha=0.2)
        ax[i].set_title(model_name)
        ax[i].set_xlabel('Number of patients')
        if i == 0:
            ax[i].set_ylabel('Difference to true ATE')
            ax[i].legend(loc=1)  
    plt.subplots_adjust(hspace=0.4)
