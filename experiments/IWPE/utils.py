import pandas as pd


def create_learning_df(results, type_of_results="classify_metrics"):
    plot_dfs = []
    for result_dict in results.values():
        # TODO: if case as for k only positive results are given
        train_f1_scores = [train_dict[type_of_results]["macro avg"]["f1-score"] for train_dict in result_dict["train_results"]]
        test_f1_scores = [train_dict[type_of_results]["macro avg"]["f1-score"] for train_dict in result_dict["test_results"]]
        train_size = result_dict["train_size"]
        plot_df = {
            "train_f1_scores": train_f1_scores,
            "test_f1_scores": test_f1_scores,
            "train_size": train_size
        }
        plot_df = pd.DataFrame(plot_df)
        plot_dfs.append(plot_df)

    mean_df = pd.concat(plot_dfs).groupby("train_size").mean()
    max_df = pd.concat(plot_dfs).groupby("train_size").max()
    max_df.columns = ["max_" + column_name for column_name in max_df.columns]
    min_df = pd.concat(plot_dfs).groupby("train_size").min()
    min_df.columns = ["min_" + column_name for column_name in min_df.columns]
    learning_df = pd.concat([mean_df, max_df, min_df], axis=1).reset_index()
    return learning_df
