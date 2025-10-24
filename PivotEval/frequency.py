
from PivotEval.SHAP_kernel import get_most_important_time_step, get_shap_rank_for_all_time_step
import numpy as np
import pandas as pd
from SimplificationMethods.ORSalgorithm.Utils.line import get_pivot_points,interpolate_points_to_line
from SimplificationMethods.Seg_Least_Square.segmentedls import solve
from SimplificationMethods.ppa.ppa import get_simplification_paa
import matplotlib.pyplot as plt

from Utils.load_data import load_dataset
import os
import ast
def extract_x_pivots(OPT,L,ts, and_y=False):
    N = len(ts)

    l = L
    i = len(ts) - 1

    opt = OPT[l, i]

    x_pivots = [opt.i]
    y_pivots = [opt.slope[0] * opt.i + opt.slope[1]]
    while opt.l > 0:
        x = opt.pre
        y = opt.slope[0] * x + opt.slope[1]
        x_pivots.append(x)
        y_pivots.append(y)
        opt = OPT[opt.l - 1, opt.pre]

    x_pivots = list(reversed(x_pivots))
    y_pivots = list(reversed(y_pivots))
    if and_y:
        return x_pivots, y_pivots
    return x_pivots



def get_xpivots_lsf(data,total_pivots):
    print("Getting LSF...")
    time_series = data


    lsf_pivots = []
    for ts in time_series:
        L = total_pivots-2
        OPT = solve(list(range(1,len(ts)+1)), ts, L)
        lsf_pivots.append(extract_x_pivots(OPT=OPT, L=L, ts=ts))
    print("Done!")

    return lsf_pivots
def extract_simplification_simp_lsf(OPT, L, ts, verbose=False):
    x_pivots, y_pivots = extract_x_pivots(OPT, L, ts, and_y=True)

    line_version = interpolate_points_to_line(ts_length=len(ts), x_selected=x_pivots, y_selected=y_pivots)

    if verbose:
        plt.plot(line_version, label="Simp")
        plt.plot(ts, label="org")
        plt.scatter(x_pivots, y_pivots, marker="x", label="simplified")
        plt.legend()
        plt.show()

    return line_version





    return list(sorted(x_pivots))
def get_full_lsf_simplification(data,total_pivots):
    print("Getting LSF...")

    time_series = data


    lsf_full = []
    for ts in time_series:
        L = total_pivots-2
        OPT = solve(list(range(0,len(ts))), ts, L)
        lsf_full.append(extract_simplification_simp_lsf(OPT=OPT, L=L, ts=ts))
    print("Done!")

    return lsf_full

def get_pivot_points_and_simp_of_paa(data, total_pivots):
    print("Getting PAA...")

    time_series = data

    paa_pivots = []
    paa_simplifications = []
    for ts in time_series:
        paa_simplification,pivot_point = get_simplification_paa(ts, num_windows=total_pivots // 2)
        paa_pivots.append(pivot_point)
        paa_simplifications.append(paa_simplification)

    print("Got PAA!")
    return paa_simplifications, paa_pivots


def get_pivot_points_and_simp_of_algo(algo, model_name,data,dataset_name, total_pivots):
    # TODO: Fix so that we limit based on num segments instead of loyalty.
    simps = np.load(f"results/{dataset_name}/data/{algo}_{model_name}_TEST_normalized.npy")


    # Get alpha needed to get num segments on average
    df = pd.read_csv(f"results/{dataset_name}/{model_name}_alpha_complexity_loyalty.csv")
    df = df[df["Type"] == algo]
    df = df[df["Num Segments"] >= total_pivots-1]
    min_num_segs= min(df["Num Segments"])
    print(min_num_segs)
    df = df[df["Num Segments"] == min_num_segs]
    min_alpha_needed = sorted(df["Alpha"].tolist())[0] # Miss match between files, need to change alpha to fit.
    # Extract the simp corresponding to this alpha
    correct_alpha = simps[np.isclose(simps["alpha"][:, 0], min_alpha_needed)]

    simplifications = correct_alpha["X"].tolist()[:len(data)]
    print(f"dataset {dataset_name} has {len(simplifications)} simplifications")

    pivot_points = []
    for ts in simplifications:
        pivot_points.append(get_pivot_points(ts, x_and_y=False))
    return pivot_points, simplifications

def get_org_data(dataset_name):
    simps = np.load(f"results/{dataset_name}/data/RDP_cnn_TEST_normalized.npy")
    correct_alpha = simps[simps["alpha"][:, 0] == 0]
    time_series = correct_alpha["X"]
    return time_series
def find_peak_rank(simplification, full_ts, rank_of_time_steps):
    min_rank = max(rank_of_time_steps) # No match means max value
    error = 10**-6
    for i, (simp, full) in enumerate(zip(simplification, full_ts)):
        if abs(simp - full) < error:
            min_rank = min(rank_of_time_steps[i], min_rank)
    return min_rank

def check_each(df, df_score_all, algo):
    # Check if (maxShap,ts[maxShap]) is in the simp
    error = 10**-4
    df[f"{algo}SimpHasXY"] = df.apply(lambda row: abs(row[f"{algo}Simp"][row["ShapIndex"]] - row[f"TS"][row["ShapIndex"]]) < error, axis=1) #
    # Chck if maxShap is a pivot
    df[f"{algo}IncludesMaxShapIndex"] = df.apply(lambda row: row["ShapIndex"] in row[f"{algo}PivotPoints"], axis=1)
    # Check if maxShap is a pivot AND (maxShap,ts[maxShap]) is in the simp
    df[f"{algo}IncludesMaxShap"] = df.apply(lambda row: row[f"{algo}IncludesMaxShapIndex"] and row[f"{algo}SimpHasXY"], axis=1)

    df_score_all[f"{algo}IncludesMaxShapIndex"] = df[f"{algo}IncludesMaxShapIndex"].mean() * 100

    df_score_all[f"{algo}Score"] = df[f"{algo}IncludesMaxShap"].mean() * 100
    df_score_all[f"{algo}Random"] = df.apply(lambda row : len(row[f"{algo}PivotPoints"]) / len(row["TS"]), axis=1).mean() * 100
    df_score_all[f"{algo}PeakRank"] = df.apply(lambda row : find_peak_rank(simplification=row[f"{algo}Simp"], full_ts=row["TS"], rank_of_time_steps=row[f"ShapRank"]), axis=1).mean()
    df_score_all[f"{algo}ExpectedPeakRank"] =  df.apply(lambda row : (len(row["TS"]) + 1) / (len(row[f"{algo}PivotPoints"]) + 1), axis=1).mean()
    df_score_all[f"{algo}AvgNrPivots"] = df.apply(lambda row : len(row[f"{algo}PivotPoints"]), axis=1).mean()

    distance_when_includes_shap = []
    for index, row in df.iterrows():
        if row[f"{algo}IncludesMaxShapIndex"]:
            dist = abs(row[f"{algo}Simp"][row["ShapIndex"]] - row[f"TS"][row["ShapIndex"]])
            distance_when_includes_shap.append(dist)

    distance_when_includes_shap = np.array(distance_when_includes_shap)
    mean_dist = 0
    if distance_when_includes_shap.size > 0:
        mean_dist = np.mean(distance_when_includes_shap)
    else:
        mean_dist = 1

    df_score_all[f"{algo}DistToYWhenMaxShapInPivot"] = mean_dist

    # Lets now do distance to maxshap_y
    df_score_all[f"{algo}DistToMaxShap"] = df.apply(lambda row:  abs(row[f"{algo}Simp"][row["ShapIndex"]] - row[f"TS"][row["ShapIndex"]]), axis=1).mean()



def check_how_often_most_important_time_step_is_pivot(model_name, dataset_name, total_pivots,recompute=False):
    org_dataset = get_org_data(dataset_name)
    org_dataset = org_dataset[:2]
    folder_shap = "PivotEval/SHAP"
    os.makedirs(folder_shap, exist_ok=True)

    file_shap = f"{dataset_name}_{model_name.split('.')[0]}.csv"
    file_shap_rank = f"{dataset_name}_{model_name.split('.')[0]}_rank.csv"
    if recompute or not os.path.exists(f"{folder_shap}/{file_shap}") or not os.path.exists(f"{folder_shap}/{file_shap_rank}"):
        max_shap_idx = []
        all_shap_ranks = []
        #for ts in org_dataset:
        #    max_shap_idx.append(get_most_important_time_step(model_name=model_name,dataset_name=dataset_name, time_series_to_be_explained=ts))
        #    all_shap_ranks.append(get_shap_rank_for_all_time_step(model_name=model_name,dataset_name=dataset_name,instance_to_be_explained=ts))
        all_ranks = get_shap_rank_for_all_time_step(model_name=model_name,dataset_name=dataset_name, background=org_dataset,instances_to_be_explained=org_dataset)
        max_shap_idx = [np.argmin(rank_ts) for rank_ts in all_ranks]
        df_shap = pd.DataFrame({"ShapIndex": max_shap_idx})
        df_shap.to_csv(folder_shap + "/"+ file_shap, index=False)
        df_shap_rank = pd.DataFrame()
        df_shap_rank["ShapRank"] = all_ranks
        df_shap_rank.to_csv(folder_shap + "/"+ file_shap_rank, index=False)



    shap_values = pd.read_csv(f"{folder_shap}/{file_shap}")["ShapIndex"].tolist()
    shap_value_ranks = pd.read_csv(f"{folder_shap}/{file_shap_rank}")["ShapRank"].apply(ast.literal_eval).tolist()

    paa_simplifications, paa_pivots = get_pivot_points_and_simp_of_paa(data=org_dataset,total_pivots=total_pivots)
    df_data = pd.DataFrame({
        "TS": [list(row) for row in org_dataset],
        "ShapIndex": shap_values,
        "ShapRank": shap_value_ranks,
        "SLSPivotPoints": get_xpivots_lsf(data=org_dataset,total_pivots=total_pivots),
        "SLSSimp": get_full_lsf_simplification(data=org_dataset,total_pivots=total_pivots),
        "PAAPivotPoints": paa_pivots,
        "PAASimp": paa_simplifications

    })

    our_algos = ["OS", "BU", "RDP","VW"]

    for algo in our_algos:
        pivots, simp = get_pivot_points_and_simp_of_algo(algo=algo,model_name=model_name, data=org_dataset,dataset_name=dataset_name, total_pivots=total_pivots)
        df_data[f"{algo}PivotPoints"] = pivots
        df_data[f"{algo}Simp"] = simp

    test_algos  =["SLS", "PAA"] # We compare to these

    df_score_all = {}
    all_algos = our_algos + test_algos
    for algo in all_algos:
        check_each(df=df_data, algo=algo, df_score_all=df_score_all)
    for algo in all_algos:
        print(f"{algo} Average Nr Pivots: ", f"{df_score_all[f'{algo}AvgNrPivots']}")

    for algo  in all_algos:
        print(f"{algo} MAX SHAP is in pivots:",f"{df_score_all[f'{algo}Score']}%", f"({round(df_score_all[f'{algo}Random'],2)}%)")

    for algo in all_algos:
        print(f"{algo} Average Rank Of Most Influential Pivot: ", f"{df_score_all[f'{algo}PeakRank']}",f"({round(df_score_all[f'{algo}ExpectedPeakRank'],2)})")

    for algo in all_algos:
        print(f"{algo} Distance To Most Influential Pivot: ", f"{df_score_all[f'{algo}DistToMaxShap']}")

    for algo in all_algos:
        print(f"{algo} Distance when max shap pivot: ", f"{df_score_all[f'{algo}DistToYWhenMaxShapInPivot']}")


    get_latex(all_algos, df_score_all, dataset_name)
    df = pd.DataFrame()
    for key, value in df_score_all.items():
        df[key] = [value]
    return df
def get_latex(algos, df_score_random,dataset_name):
    # Generate LaTeX table

    latex_table = []

    org_algos = ["RDP", "OS", "VW","BU"]
    compare_algos = ["SLS", "PAA"] # Fix rekkefølge på disse på MAX SHAP
    algos = org_algos + compare_algos
    latex_table.append("\\begin{tabular}{|l|c|c|c|c|}")
    latex_table.append("\\hline")
    latex_table.append("Algo & MAX SHAP $\\uparrow$ & CorrectX $\\uparrow$ & Dist $\\downarrow$ & Rank $\\downarrow$ \\\\")
    latex_table.append("\\hline")

    max_shap = max([round(df_score_random[f'{algo}Score'],2) for algo in algos])
    min_rank = min([round(df_score_random[f'{algo}PeakRank'],2) for algo in algos])
    min_distance = min([round(df_score_random[f'{algo}DistToMaxShap'],2) for algo in algos])
    for algo in algos:
        if algo == "SLS":
            latex_table.append("\\hline")

        shap_score = round(df_score_random[f'{algo}Score'],2)
        rank_score = round(df_score_random[f'{algo}PeakRank'],2)

        dist_score = round(df_score_random[f'{algo}DistToYWhenMaxShapInPivot'],2)

        correctX = round(df_score_random[f"{algo}IncludesMaxShapIndex"],2)

        # Format the SHAP column to include both values

        shap_column = f"${shap_score}\\%$"
        if max_shap == shap_score:
            shap_column = "$\\textbf{" + str(shap_score) + "}\\%$"
        # Format the rank column to include both values
        rank_column = f"${rank_score}$"
        if min_rank == rank_score:
            rank_column = "$\\textbf{" + str(rank_score) + "}$"

        dist_column = f"${dist_score}$"
        if dist_score == min_distance:
            dist_column = "$\\textbf{" + str(dist_score) + "}$"

        correctX_column = f"${correctX}\\%$"




        latex_table.append(f"{algo} & {shap_column} & {correctX_column}  & {dist_column}  & {rank_column} \\\\")

    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")


    # Join all lines and print the LaTeX code
    latex_code = "\n".join(latex_table)
    print("LaTeX Table Code:")
    print(latex_code)
    folder = "PivotEval"
    path = f'{folder}/algorithm_table_{dataset_name}.tex'
    # Save to file
    with open(path, 'w') as f:
        f.write(latex_code)

    print(f"\nLaTeX table saved to {path}")

def aggregate_over_all_dfs(list_of_dfs):
    mean_df = sum(list_of_dfs) / len(list_of_dfs)
    return mean_df

if __name__ == "__main__":
    total_pivots = {
        "SonyAIBORobotSurface1":12,
        "Chinatown":7,
        "ECG200":9,
        "DistalPhalanxOutlineAgeGroup":12

    }
    dataset_names = ["ECG200"]#list(total_pivots.keys())


    model_name = "miniRocket.pkl"
    redo = True
    recompute = True
    all_dfs = []
    for dataset_name in dataset_names:
        folder = "PivotEval"
        path = f'{folder}/{dataset_name}_df.csv'
        if redo:
            curr_pivots = total_pivots[dataset_name]
            df = check_how_often_most_important_time_step_is_pivot(dataset_name=dataset_name, model_name=model_name, recompute=recompute, total_pivots=curr_pivots)

            df.to_csv(path, index=False)
            all_dfs.append(df)
        else:
            curr_df = pd.read_csv(path)
            all_dfs.append(curr_df)

        #get_pivot_points_of_lsf(dataset_name=dataset_name)

    over_all_datasets_df = aggregate_over_all_dfs(all_dfs)
    algos = ["OS", "BU", "RDP","VW", "SLS", "PAA"] # Fix rekkefølge på disse på MAX SHAP
    dict_all= over_all_datasets_df.to_dict('records')[0]

    get_latex(algos=algos, df_score_random=dict_all, dataset_name="over_all")