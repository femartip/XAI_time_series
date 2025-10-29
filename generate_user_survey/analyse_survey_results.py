import pandas as pd
import numpy as np
import re
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

SEED = 42
warnings.filterwarnings('ignore', category=RuntimeWarning)

def bootstrap_ci(data, ):
    """Using nonparametric bootstrap resampling"""
    data = np.asarray(data)     #Each mean acc is one observation
    rng = default_rng(SEED)     
    boot_means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(2000)]     #Mean acc of resampled observations over 2000 iterations
    lower, upper = np.quantile(boot_means, [0.05/2, 1 - 0.05/2])    # 95% confidence interval
    return lower, upper

def violin_plot_per_participant(accs_by_level, levels, title, ylabel):
    rows = []
    for lvl, accs in zip(levels, accs_by_level):
        for a in accs:
            rows.append({"Level": lvl, "ParticipantAccuracy": a})
    dfv = pd.DataFrame(rows)

    plt.figure(figsize=(8,6))
    ax = sns.violinplot(
        data=dfv, x="Level", y="ParticipantAccuracy",
        order=levels, cut=0, inner="box"
    )
    sns.stripplot(
        data=dfv, x="Level", y="ParticipantAccuracy",
        order=levels, dodge=False, jitter=0.08, alpha=0.6, ax=ax
    )
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Loyalty")
    plt.tight_layout()
    plt.savefig(f"./generate_user_survey/results/{dataset}_violin_plot.png")
    
    

def analyze_survey_results(dataset: str) -> pd.DataFrame:
    df_raw = pd.read_csv(f"./generate_user_survey/results/{dataset}.csv", header=None)

    loyalties = df_raw.iloc[0].dropna().tolist()
    def parse_loyalty(x):
        m = re.search(r"L(\d+)", x)
        return float(m.group(1))/100 if m else 1.00
    loyalty_vals = [parse_loyalty(x) for x in loyalties]

    df = df_raw.iloc[1:].reset_index(drop=True)

    cols_per_level = df.shape[1] // 4
    level_blocks = np.array_split(df.columns, 4)

    results = []
    accs_by_level = []
    for lvl, cols, loyalty in zip(loyalties, level_blocks, loyalty_vals):
        sub = df[cols]
        #print(f"Processing Level: {lvl} with Loyalty: {loyalty}")
        #print(sub)
        sub_no_letters = sub[1:][:]
        
        list_answers = []
        accuracy_list = []
        for idx, cols in sub_no_letters.items():
            answers = cols.tolist()
            answers = [int(x) for x in answers]
            ## Careful, I am assuming that all-zero answers are invalid and should be skipped
            if answers == [0]*len(answers):
                continue
            list_answers.append(answers)
            accuracy_list.append(sum(answers)/len(answers))

        flattened_answers = [val for answers in list_answers for val in answers]
        accs_by_level.append(accuracy_list)
        lo, hi = bootstrap_ci(accuracy_list)
        arr = np.array(list_answers)  # shape (participants, items)
        
        corr_matrix = np.corrcoef(arr)      # Pearson correlation between participants
        mean_r = np.nanmean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        acc = sum(accuracy_list)/len(accuracy_list) 
        delta = acc - loyalty
        #nlift = delta / (1 - loyalty) if loyalty < 1 else np.nan

        results.append({
            "Level": lvl,
            "Mean_Accuracy": acc,
            "CI_Lower": lo,
            "CI_Upper": hi,
            "Delta_CI": hi-lo,
            "Mean_Interparticipant_Correlation": mean_r,
            "Delta_vs_Loyalty": delta,
            #"Nlift_vs_Loyalty": nlift,
        })

    violin_plot_per_participant(accs_by_level, loyalties,title=f"{dataset}",ylabel="Participant Accuracy")
    summary = pd.DataFrame(results)
    print(f"Survey Analysis Summary for {dataset}:")
    print(summary)
    return summary


if __name__ == "__main__":
    datasets = ["Chinatown", "ECG200", "SonyAIBORobotSurface1", "UMD"]
    df_all_datasets = []
    for dataset in datasets:
        df = analyze_survey_results(dataset)
        df_all_datasets.append(df)

    df_final = pd.concat(df_all_datasets, keys=datasets, names=["Dataset"])
    df_final.to_csv("./generate_user_survey/results/survey_analysis_summary.csv")
