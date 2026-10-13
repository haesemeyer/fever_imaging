"""
Script to co-cluster activity from PBS and polyIC injected fish followed by analysis of per-type cluster identities.
"""

import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib as mpl
import argparse
import os
from os import path
import seaborn as sns
from scipy.stats import ranksums, rankdata
from scipy.ndimage import gaussian_filter1d
import utilities
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.signal import butter, sosfiltfilt


def compute_mm_delta_temps(temps: np.ndarray, frequency_hz: int) -> np.ndarray:
    """
    Tries to approximate the "previous bout delta-T" in the Markov Model by assigning the change across
    the previous second to each timepoint
    :param temps: The temperatures for which to compute delta-T
    :param frequency_hz: The samples per second for the temps recording
    :return: Matched prev-second delta-T
    """
    delta_temps = np.full_like(temps, np.nan)
    for ix in range(delta_temps.size):
        if ix == 0:
            delta_temps[ix] = 0
        elif ix < frequency_hz:
            delta_temps[ix] = (temps[ix] - temps[0]) * frequency_hz / ix  # extrapolate
        else:
            delta_temps[ix] = temps[ix] - temps[ix - frequency_hz]
    return delta_temps


def tm_from_glm(t: float, dt: float, glm_orig_coef: np.ndarray, glm_icept: np.ndarray) -> pd.DataFrame:
    """
    From temperature and delta-temp inputs uses the glm coefficients to compute the transition matrix
    :param t: The temperature at which to compute the transition matrix
    :param dt: The delta-temperature at which to compute the transition matrix
    :param glm_orig_coef: The glm coefficients of the Markov Model transition
    :param glm_icept: The glm intercepts of the Markov Model transition
    :return: The transition matrix
    """
    trans_mat = np.zeros((3, 3))
    lp = np.zeros(3)

    design = np.array([t, dt, t*dt, t**2, dt**2, t*t*dt, t*dt*dt, t**3, dt**3])
    # Transitions from reversal
    lp[0] = glm_orig_coef[0, :, 0] @ design + glm_icept[0, 0]
    lp[1] = glm_orig_coef[0, :, 1] @ design + glm_icept[0, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[0, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from general
    lp[0] = glm_orig_coef[1, :, 0] @ design + glm_icept[1, 0]
    lp[1] = glm_orig_coef[1, :, 1] @ design + glm_icept[1, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[1, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from persistent
    lp[0] = glm_orig_coef[2, :, 0] @ design + glm_icept[2, 0]
    lp[1] = glm_orig_coef[2, :, 1] @ design + glm_icept[2, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[2, k] = exp_lp[k] / np.sum(exp_lp)
    return pd.DataFrame(trans_mat, index=["R(o)", "X(o)", "P(o)"], columns=["R(i)", "X(i)", "P(i)"])


def lineplot(data: pd.DataFrame, y: str, hue: Optional[str], x: np.ndarray, x_name: str, boot_fun: np.mean, nboot=1000,
             line_args: Optional[Dict] = None, shade_args: Optional[Dict] = None, add_marker=False) -> pl.Figure:
    """
    Seaborn style lineplot function that does not require x-values to be within the dataframe
    :param data: Dataframe with the plot-data
    :param y: Name of the column with y-values within the dataframe
    :param hue: Name of the column to split observations
    :param x: vector of x-values with same length as y-values
    :param x_name: The label of the x-axis
    :param boot_fun: The function to use for bootstrapping
    :param nboot: The number of bootstrap samples to generate
    :param line_args: Additional keyword arguments for the boot-average lineplot
    :param shade_args: Additional keyword arguments for the boot-error shading
    :param add_marker: If set each bin center will be marked
    :return: The generated figure object
    """
    if shade_args is None:
        shade_args = {}
    if line_args is None:
        line_args = {}
    if 'alpha' not in shade_args:
        shade_args['alpha'] = 0.4
    if hue is None:
        hues = [None]
    else:
        hues = np.unique(data[hue])
    fig = pl.figure()
    for i, h in enumerate(hues):
        if h is None:
            values = np.vstack(data[y])
        else:
            values = np.vstack(data[y][data[hue] == h])
        boot_variate = utilities.bootstrap(values, nboot, boot_fun)
        m = np.mean(boot_variate, axis=0)
        e = np.std(boot_variate, axis=0)
        if 'color' not in shade_args:
            pl.fill_between(x, m - e, m + e, color=f"C{i}", **shade_args)
        else:
            pl.fill_between(x, m - e, m + e, **shade_args)
        if 'color' not in line_args:
            pl.plot(x, m, f"C{i}", label=h, marker='.' if add_marker else "None", **line_args)
        else:
            pl.plot(x, m, label=h, marker='.' if add_marker else "None", **line_args)
    pl.xlabel(x_name)
    pl.ylabel(f"{y} +/- se")
    pl.legend()
    sns.despine()
    return fig


def load_data_group(hdf_dir: str) -> Dict:
    hdf_files = [path.join(hdf_dir, f) for f in os.listdir(hdf_dir) if ".hdf5" in f and "cluster" not in f]
    exp_folders = [f[:-5]+"/" for f in hdf_files]
    temp_files = [path.join(ef, f) for ef in exp_folders for f in os.listdir(ef) if ".temp" in f and "cluster" not in f]

    all_temperatures = [np.genfromtxt(tf) for tf in temp_files]
    all_temperatures = [at for at in all_temperatures if at.size > 20000]
    min_len = min([len(at) for at in all_temperatures])

    # collect and concatenate all responses
    all_temperature_neurons = []
    all_temperature_spikes = []
    all_indices = []
    all_test_correlations = []
    all_plane_ids = []  # simple set of ids where all neurons from one plane receive the same consecutive number
    all_fish_ids = []
    plane_counter = 0
    for fid, f in enumerate(hdf_files):
        with h5py.File(f, "r") as dfile:
            for k in dfile:
                if "weights" in k:
                    continue
                grp = dfile[k]
                responses = grp["heat_res_neurons"][()]
                if responses.shape[1] != 5400:
                    print(f"In plane {k} only {responses.shape[1]} timepoints were recorded.")
                    continue
                try:
                    spks = grp["heat_res_spikes"][()]
                except KeyError:
                    spks = responses.copy()
                indices = grp["heat_res_indices"][()]
                test_corrs = grp["heat_res_testcorr"][()]
                all_test_correlations.append(test_corrs)
                # filter by test correlation - rather loosely, let clustering figure out the rest
                responses = responses[test_corrs > 0.6]
                spks = spks[test_corrs > 0.6]
                indices = indices[test_corrs > 0.6]
                all_temperature_neurons.append(responses)
                all_temperature_spikes.append(spks)
                all_indices.append(indices)
                all_plane_ids.append(np.full(responses.shape[0], plane_counter))
                all_fish_ids.append(np.full(responses.shape[0], fid))
                plane_counter += 1
    all_temperature_neurons = np.vstack(all_temperature_neurons)[:, :-1]
    all_temperature_spikes = np.vstack(all_temperature_spikes)[:, :-1]
    all_temperature_neurons = utilities.safe_standardize(all_temperature_neurons, 1)
    all_plane_ids = np.hstack(all_plane_ids)
    all_test_correlations = np.hstack(all_test_correlations)
    all_fish_ids = np.hstack(all_fish_ids)
    all_indices = np.hstack(all_indices)
    return {
        "z_scored_fluorescence": all_temperature_neurons,
        "deconvolved_fluorescence": all_temperature_spikes,
        "plane_ids": all_plane_ids,
        "fish_ids": all_fish_ids,
        "indices": all_indices,
        "test_corrs": all_test_correlations,
        "temperatures": np.vstack([temps[:min_len] for temps in all_temperatures])
    }


def plot_cluster_error(data, xvals, cluster, treatment, ax, color, legend=None) -> int:
    cluster_act = data["Activity"][np.logical_and(data["Treatment"] == treatment, data["Cluster"] == cluster)]
    if cluster_act.size == 0:
        return 0
    cluster_act = np.vstack(cluster_act)
    boot_result = utilities.bootstrap(cluster_act, 100, np.mean)
    average = np.mean(boot_result, axis=0)
    error = np.std(boot_result, axis=0)
    ax.fill_between(xvals, average - error, average + error, color=color, alpha=0.4)
    ax.plot(xvals, average, color=color, label=treatment if legend is None else legend)
    return cluster_act.shape[0]


def mse_bs_hyp_test(data: pd.DataFrame, cluster: int, nboot=1000) -> Tuple[float, float]:
    """
    Performs a bootstrap hypothesis test across fish for a given cluster based on the mean squared difference in traces
    :param data: Per fish cluster averages for each treatment
    :param cluster: The cluster number for which to perform the test
    :param nboot: The number of bootstrap iterations to perform
    :return:
        [0]: The bootstrap p-value
        [1]: The true mean-squared distance
    """
    act_control = np.vstack(data["Activity"][np.logical_and(data["Treatment"] == "PBS Control", data["Cluster"] == cluster)])
    act_poly = np.vstack(data["Activity"][np.logical_and(data["Treatment"] == "poly(I:C)", data["Cluster"] == cluster)])
    mse_real = np.mean((np.mean(act_control, 0) - np.mean(act_poly, 0))**2)
    cnt_control = act_control.shape[0]
    act_combined = np.vstack([act_control, act_poly])
    ix_combined = np.arange(act_combined.shape[0]).astype(int)
    mse_boot = np.zeros(nboot)
    for i in range(nboot):
        ix_boot = np.random.choice(ix_combined, size=ix_combined.size, replace=True)
        boot_control = act_combined[ix_boot[:cnt_control]]
        boot_poly = act_combined[ix_boot[cnt_control:]]
        mse_boot[i] = np.mean((np.mean(boot_control, 0) - np.mean(boot_poly, 0))**2)
    return (1+np.sum(mse_boot > mse_real)) / (nboot+1), mse_real


def cluster_by_exemplars(neuron_data: np.ndarray, exemplars: np.ndarray, corrcut: float):
    """
    Assigns clusters to each neuron in neuro_data based on proximity to cluster center examples measured by correlation
    :param neuron_data: n_neurons x m_timepoints array of neuronal responses
    :param exemplars: n_timepoints x m_clusters array of cluster centroids
    :param corrcut: The cutoff in terms of correlation to decide whether a neuron belongs into the cluster of a given centroid
    :return: n_neurons long vector of cluster assignments
    """
    corr_mat = np.zeros((neuron_data.shape[0], exemplars.shape[1]))
    for i, neuron in enumerate(neuron_data):
        for j, exemplar in enumerate(exemplars.T):
            corr_mat[i, j] = np.corrcoef(neuron, exemplar)[0, 1]
    ix_max = np.argmax(corr_mat, axis=1)
    val_max = np.max(corr_mat, axis=1)
    clust_assign = np.full(neuron_data.shape[0], -1)
    clust_assign[val_max >= corrcut] = ix_max[val_max >= corrcut]
    return clust_assign


def iterate_find_predictor(init_exemplars: np.ndarray, neuron_data: np.ndarray, corrcut: float, n_iter: int):
    """
    Starting from an initial set of seeds converges onto the most likely cluster center corresponding to the seed
    :param init_exemplars: n_timepoints x m_clusters array of cluster seeds
    :param neuron_data: n_neurons x m_timepoints array of neuronal responses in which to find the true cluster centers
    :param corrcut: The cutoff to consider a neuron's activity to be in the cluster of the current center
    :param n_iter: The number of iterations to run
    :return: Standardized version of the true cluster centers (n_timepoints x m_clusters)
    """
    exemplars = init_exemplars.copy()
    for i in range(n_iter):
        ex_new = exemplars.copy()
        assignments = cluster_by_exemplars(neuron_data, exemplars, corrcut)
        for j in range(init_exemplars.shape[1]):
            if np.sum(assignments == j) > 20:# was 10
                ex_new[:, j] = np.mean(neuron_data[assignments == j], axis=0)
        iter_corrs = [np.corrcoef(new, old)[0, 1] for new, old in zip(ex_new.T, exemplars.T)]
        if min(iter_corrs) > 0.99:
            print(f"Convergence in iteration {i}")
            exemplars = ex_new
            break
        exemplars = ex_new
    return utilities.safe_standardize(exemplars, axis=0)


def compute_ranksums(data: pd.DataFrame, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute ranksum based comparisons for the specific activity dataframes
    """
    pv = []
    counts = []
    for cid in clus_nums:
        d_pbs = data[y_name][np.logical_and(data["Treatment"]=="PBS Control",
                                                 data["Response type"]==cluster_names[cid])]
        d_poly = data[y_name][np.logical_and(data["Treatment"]=="poly(I:C)",
                                                 data["Response type"]==cluster_names[cid])]
        p = ranksums(d_pbs, d_poly).pvalue
        pv.append(p)
        counts.append([d_pbs.shape[0], d_poly.shape[0]])
    return np.array(pv), np.vstack(counts)

def generate_stat_df(p_values: np.ndarray, counts: np.ndarray, names: np.ndarray):
    """
    For a set of p-values generates a dataframe that allows comparing of the p-values
    to critical values for Benjamini-Hochberg at different FDRs
    :param p_values: For each individual comparison the p-value
    :param counts: For each individual comparison the count of both items (n_items * 2)
    :param names: The names of the conditions that were compared
    :return:
    """
    pv_ranks = rankdata(p_values, method='max')
    rank_ix = np.argsort(pv_ranks)
    fdrs = [0.05, 0.01, 0.001]
    data = {"Name": [], "n": [], "p-value": [], "Rank": []} | {f"Critical at {q}": [] for q in fdrs}
    for ix in rank_ix:
        data["Name"].append(names[ix])
        data["n"].append(counts[ix])
        data["p-value"].append(p_values[ix])
        data["Rank"].append(pv_ranks[ix])
        for q in fdrs:
            data[f"Critical at {q}"].append(pv_ranks[ix]/p_values.size*q)
    return pd.DataFrame(data)


if __name__ == '__main__':

    mpl.rcParams['pdf.fonttype'] = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="fever_cluster_analysis",
                                       description="Clusters flow setup temperature responses for behavioral fever data")
    a_parser.add_argument("-dc", "--control", help="Path to folder with response Hdf5 file of pbs control",
                          type=str, required=True)
    a_parser.add_argument("-df", "--fever", help="Path to folder with response Hdf5 file of polyIC",
                          type=str, required=True)

    args = a_parser.parse_args()

    hdf_dir_control = path.normpath(args.control)
    hdf_dir_fever = path.normpath(args.fever)

    control_data = load_data_group(hdf_dir_control)
    fever_data = load_data_group(hdf_dir_fever)

    analysis_dir = path.split(hdf_dir_fever)[0]

    medulla_types = np.load(path.join(analysis_dir, "KAB_medulla_cluster_averages.npy"))

    # generate per-treatment response archetypes from the original medulla predictors
    control_neurons = control_data["z_scored_fluorescence"].copy()
    fever_neurons = fever_data["z_scored_fluorescence"].copy()

    fever_medulla = iterate_find_predictor(medulla_types, fever_neurons, np.sqrt(0.5), 100)
    cont_medulla = iterate_find_predictor(medulla_types, control_neurons, np.sqrt(0.5), 100)

    # filter neuron responses
    sos = butter(4, 0.2, 'low', fs=5, output='sos')
    control_neurons = sosfiltfilt(sos, control_neurons, axis=1)
    fever_neurons = sosfiltfilt(sos, fever_neurons, axis=1)
    # assign clusters per treatment
    clus_control = cluster_by_exemplars(control_neurons, cont_medulla, 0.6)
    clus_fever = cluster_by_exemplars(fever_neurons, fever_medulla, 0.6)

    cluster_membership = np.hstack([clus_control, clus_fever])

    # merge data across experiments
    all_neurons = np.vstack([control_neurons, fever_neurons])
    all_fish_ids = np.hstack([control_data["fish_ids"], fever_data["fish_ids"]+100])
    min_stim_length = min([control_data["temperatures"].shape[1], fever_data["temperatures"].shape[1]])
    avg_stimulus = np.mean(np.vstack([control_data["temperatures"][:, :min_stim_length],
                                      fever_data["temperatures"][:, :min_stim_length]]), axis=0)
    data_type = np.hstack([np.zeros(control_data["test_corrs"].size), np.ones(fever_data["test_corrs"].size)])

    clus_nums = np.unique(cluster_membership[cluster_membership > -1])

    # compute cluster averages per fish
    ind_averages = {"Activity": [], "Treatment": [], "Cluster": []}
    for fid in np.unique(all_fish_ids):
        this_fish_clusters = cluster_membership[all_fish_ids == fid]
        this_fish_activity = all_neurons[all_fish_ids == fid]
        for c in clus_nums:
            if np.any(this_fish_clusters == c):
                ind_averages["Cluster"].append(c)
                if fid < 100:
                    ind_averages["Treatment"].append("PBS Control")
                else:
                    ind_averages["Treatment"].append("poly(I:C)")
                ind_averages["Activity"].append(np.mean(this_fish_activity[this_fish_clusters == c], axis=0))
    df_ind_averages = pd.DataFrame(ind_averages)

    stim_times = np.arange(avg_stimulus.size) / 20

    plot_act = []
    for cid in clus_nums:
        # control
        plot_prob = 50 / np.sum(clus_control == cid)
        to_plot = np.random.rand(np.sum(clus_control == cid)) < plot_prob
        plot_act.append(control_neurons[clus_control == cid][to_plot])
        # treatment spacer
        plot_act.append(np.full((2, control_neurons.shape[1]), np.nan))
        # fever
        plot_prob = 50 / np.sum(clus_fever == cid)
        to_plot = np.random.rand(np.sum(clus_fever == cid)) < plot_prob
        plot_act.append(fever_neurons[clus_fever == cid][to_plot])
        # type spacer
        plot_act.append(np.full((5, control_neurons.shape[1]), np.nan))

    neuron_time = np.arange(all_neurons.shape[1]) / 5

    plot_act = np.vstack(plot_act)
    df_plot_act = pd.DataFrame(plot_act, index=np.arange(plot_act.shape[0]), columns=neuron_time)

    fig = pl.figure()
    sns.heatmap(df_plot_act, vmax=5, rasterized=True, cmap='viridis', xticklabels=300)
    pl.xlabel("Time [s]")
    pl.ylabel("Sorted neurons")
    fig.savefig(path.join(analysis_dir, "clustered_heatmap.pdf"), dpi=300)

    cluster_names = {0: "CA", 1: "H", 2: "HC", 3: "C", 4: "CC", 5: "HH", 6: "CH"}

    # Plot per-fish cluster counts
    cluster_counts = {"Response type": [], "Count": [], "Treatment": []}
    for fid in np.unique(all_fish_ids):
        for cid in clus_nums:
            if np.sum(cluster_membership[all_fish_ids==fid] == cid) == 0:
                continue
            cluster_counts["Response type"].append(cluster_names[cid])
            cluster_counts["Count"].append(np.sum(cluster_membership[all_fish_ids==fid] == cid))
            cluster_counts["Treatment"].append("PBS Control" if np.all(data_type[all_fish_ids==fid] == 0) else "poly(I:C)")

    cluster_counts = pd.DataFrame(cluster_counts)

    fig = pl.figure()
    sns.boxplot(data=cluster_counts, x="Response type", y="Count", hue="Treatment", whis=np.inf)
    sns.despine()
    pl.yscale('log')
    fig.savefig(path.join(analysis_dir, "fish_cluster_membership_count.pdf"))

    # save significance data for cluster counts
    cc_pvals, cc_counts = compute_ranksums(cluster_counts, "Count")
    cnt_sig_df = generate_stat_df(cc_pvals, cc_counts,
                                  np.array([cluster_names[i] for i in range(clus_nums.size)]))
    cnt_sig_df.to_csv(path.join(analysis_dir, "stats_cluster_counts.csv"), index=False)

    # Plot fraction of fish with given cluster
    cluster_fraction = {"Response type": [], "Fraction": [], "Treatment": []}
    fish_count_control = np.unique(all_fish_ids[all_fish_ids < 100]).size
    fish_count_poly = np.unique(all_fish_ids[all_fish_ids >= 100]).size
    for cid in clus_nums:
        cluster_fish = all_fish_ids[cluster_membership == cid]
        cont_fish_cnt = np.unique(cluster_fish[cluster_fish < 100]).size
        poly_fish_cnt = np.unique(cluster_fish[cluster_fish >= 100]).size
        cluster_fraction["Response type"].append(cluster_names[cid])
        cluster_fraction["Response type"].append(cluster_names[cid])
        cluster_fraction["Treatment"].append("PBS")
        cluster_fraction["Fraction"].append(cont_fish_cnt / fish_count_control)
        cluster_fraction["Treatment"].append("Poly(I:C)")
        cluster_fraction["Fraction"].append(poly_fish_cnt / fish_count_poly)

    fig = pl.figure()
    sns.barplot(data=cluster_fraction, x="Response type", y="Fraction", hue="Treatment")
    sns.despine()
    fig.savefig(path.join(analysis_dir, "fish_cluster_fraction.pdf"))

    cluster_baseline_periods = {6: (150, 500), 5: (350, 600), 4: (550, 850), 3: (150, 300), 2: (450, 700),
                                1: (0, 150), 0: (550, 850)}
    # Depending on the type we are either interested in the minimal or maximal temperature at which the type crosses
    # threshold
    cluster_temp_of_interest = {0: np.max, 1: np.min, 2: np.max, 3: np.min, 4: np.max, 5: np.min, 6: np.min}

    cluster_baseline_m = {}
    cluster_baseline_s = {}
    act_p_values = np.full(clus_nums.size, np.nan)
    for i, cn in enumerate(clus_nums):
        cluster_act = np.mean(df_ind_averages["Activity"][df_ind_averages["Cluster"] == cn], axis=0)
        start = np.argmin(np.abs(neuron_time - cluster_baseline_periods[cn][0]))
        end = np.argmin(np.abs(neuron_time - cluster_baseline_periods[cn][1]))
        cluster_baseline_m[cn] = np.mean(cluster_act[start:end])
        cluster_baseline_s[cn] = np.std(cluster_act[start:end])
        act_p_values[i] = mse_bs_hyp_test(df_ind_averages, cn, 10000)[0]

    neuron_counts = []
    for i, cn in enumerate(clus_nums):
        p_legend = f"poly(I:C) p={act_p_values[i]:.2g}"
        fig, ax = pl.subplots()
        plot_cluster_error(df_ind_averages, neuron_time, cn, "PBS Control", ax, "C0")
        plot_cluster_error(df_ind_averages, neuron_time, cn, "poly(I:C)", ax, "C1", legend=p_legend)
        pl.plot(neuron_time, np.ones_like(neuron_time) * cluster_baseline_m[cn], "C3", alpha=0.5)
        pl.plot(neuron_time, np.ones_like(neuron_time) * cluster_baseline_m[cn] + 3*cluster_baseline_s[cn], "C3", alpha=0.5)
        pl.xlabel("Time [s]")
        pl.ylabel("Activity [AU]")
        pl.title(f"Cluster {cluster_names[cn]}")
        ax_op = pl.twinx()
        ax_op.plot(stim_times, avg_stimulus, 'k--')
        ax_op.set_ylabel("Temperature [C]")
        ax.legend()
        fig.savefig(path.join(analysis_dir, f"fish_cluster_{cluster_names[cn]}_activity.pdf"))

    # save significance data for activity
    neuron_counts = []
    for i in range(clus_nums.size):
        name = cluster_names[clus_nums[i]]
        neuron_counts.append(np.array([
            df_ind_averages["Activity"][np.logical_and(df_ind_averages["Treatment"] == "PBS Control", df_ind_averages["Cluster"] == clus_nums[i])].shape[0],
            df_ind_averages["Activity"][np.logical_and(df_ind_averages["Treatment"] == "poly(I:C)", df_ind_averages["Cluster"] == clus_nums[i])].shape[0]
                             ]))
    act_sig_df = generate_stat_df(act_p_values, np.vstack(neuron_counts), np.array([cluster_names[i] for i in range(clus_nums.size)]))
    act_sig_df.to_csv(path.join(analysis_dir, "stats_cluster_activity.csv"), index=False)

    def find_activation_ix(activity: np.ndarray, threshold: float, pk_threshold: float):
        # Identify activity threshold crossing that lead to larger peaks in responses - many types have small side-peaks
        # at low temperatures and the meaning of these is unclear so we want to exclud those.
        abt = activity > threshold
        abp = activity > pk_threshold
        crossing = np.r_[0, np.diff(abt)]
        crossing_pk = np.r_[0, np.diff(abp)]
        cross_indices = np.where(crossing > 0)[0]
        pk_indices = np.where(crossing_pk > 0)[0]
        crossings = np.zeros(pk_indices.size, dtype=int)
        for i, pi in enumerate(pk_indices):
            # for each peak index find the closest cross index *before* the peak index
            diffs = (pi - cross_indices).astype(float)
            diffs[diffs < 0] = np.inf
            crossings[i] = cross_indices[np.argmin(diffs)]
        return crossings


    temperature_interp = np.interp(neuron_time, stim_times, avg_stimulus)[:, None]

    d_activation_temps = {"Treatment": [], "Response type": [], "Threshold temperature": []}
    for row in df_ind_averages.iterrows():
        cn = row[1]["Cluster"]
        # compute baseline for this fish and cluster
        start = np.argmin(np.abs(neuron_time - cluster_baseline_periods[cn][0]))
        end = np.argmin(np.abs(neuron_time - cluster_baseline_periods[cn][1]))
        baseline_m = np.mean(row[1]["Activity"][start:end])
        baseline_s = np.std(row[1]["Activity"][start:end])
        ix_activation = find_activation_ix(row[1]["Activity"], baseline_m + 3*baseline_s, baseline_m + 10*baseline_s)
        if len(ix_activation) == 0:
            continue
        d_activation_temps["Treatment"].append(row[1]["Treatment"])
        d_activation_temps["Response type"].append(cluster_names[cn])
        d_activation_temps["Threshold temperature"].append(cluster_temp_of_interest[cn](temperature_interp[ix_activation]))

    df_activation_temps = pd.DataFrame(d_activation_temps)
    fig = pl.figure()
    sns.boxplot(data=df_activation_temps, x="Response type", y="Threshold temperature", hue="Treatment", whis=np.inf)
    sns.despine()
    fig.savefig(path.join(analysis_dir, "responsetype_activation_temps.pdf"))

    # save significance data for cluster activation
    ca_pvals, ca_counts = compute_ranksums(df_activation_temps, "Threshold temperature")
    cnt_sig_df = generate_stat_df(ca_pvals, ca_counts,
                                  np.array([cluster_names[i] for i in range(clus_nums.size)]))
    cnt_sig_df.to_csv(path.join(analysis_dir, "stats_responsetype_activation.csv"), index=False)

    # Compare neuron linearities and fit test scores to assess overall similarities/changes in medullary encoding
    lin_scores_control = np.load(path.join(hdf_dir_control, "all_lin_scores.npy"))
    lin_scores_fever = np.load(path.join(hdf_dir_fever, "all_lin_scores.npy"))
    sq_scores_control = np.load(path.join(hdf_dir_control, "all_sq_scores.npy"))
    sq_scores_fever = np.load(path.join(hdf_dir_fever, "all_sq_scores.npy"))
    d_linearity = {"Treatment": [], "Cumulative density": []}
    d_quadratic = {"Treatment": [], "Cumulative density": []}
    d_test_corrs = {"Treatment": [], "Cumulative density": []}
    score_bins = np.linspace(-1, 1, 100)
    score_bc = score_bins[:-1] + np.diff(score_bins)/2
    # Assign to fish
    f_lin_s_control, f_sq_s_control, f_corr_control = [], [], []
    for fid in np.unique(control_data["fish_ids"]):
        ls = lin_scores_control[control_data["fish_ids"] == fid]
        d_linearity["Treatment"].append("PBS Control")
        d_linearity["Cumulative density"].append(np.cumsum(np.histogram(ls, bins=score_bins, density=True)[0]))
        lsq = sq_scores_control[control_data["fish_ids"] == fid]
        d_quadratic["Treatment"].append("PBS Control")
        d_quadratic["Cumulative density"].append(np.cumsum(np.histogram(lsq, bins=score_bins, density=True)[0]))
        f_lin_s_control.append(ls)
        f_sq_s_control.append(lsq)
        tc = control_data["test_corrs"][control_data["fish_ids"] == fid]
        f_corr_control.append(tc)
        d_test_corrs["Treatment"].append("PBS Control")
        d_test_corrs["Cumulative density"].append(np.cumsum(np.histogram(tc, bins=score_bins, density=True)[0]))
    f_lin_s_fever, f_sq_s_fever, f_corr_fever = [], [], []
    for fid in np.unique(fever_data["fish_ids"]):
        ls = lin_scores_fever[fever_data["fish_ids"] == fid]
        d_linearity["Treatment"].append("poly(I:C)")
        d_linearity["Cumulative density"].append(np.cumsum(np.histogram(ls, bins=score_bins, density=True)[0]))
        f_lin_s_fever.append(ls)
        lsq = sq_scores_fever[fever_data["fish_ids"] == fid]
        d_quadratic["Treatment"].append("poly(I:C)")
        d_quadratic["Cumulative density"].append(np.cumsum(np.histogram(lsq, bins=score_bins, density=True)[0]))
        f_sq_s_fever.append(lsq)
        tc = fever_data["test_corrs"][fever_data["fish_ids"] == fid]
        f_corr_fever.append(tc)
        d_test_corrs["Treatment"].append("poly(I:C)")
        d_test_corrs["Cumulative density"].append(np.cumsum(np.histogram(tc, bins=score_bins, density=True)[0]))

    d_linearity = pd.DataFrame(d_linearity)
    d_quadratic = pd.DataFrame(d_quadratic)

    fig = lineplot(d_linearity, "Cumulative density", "Treatment", score_bc, "Linear approximation R2", np.mean)
    sns.despine()
    fig.savefig(path.join(analysis_dir, "Linearity_score_comparison.pdf"))

    fig = lineplot(d_quadratic, "Cumulative density", "Treatment", score_bc, "2nd order approximation R2", np.mean)
    sns.despine()
    fig.savefig(path.join(analysis_dir, "Quadratic_score_comparison.pdf"))

    fish_counts = np.array([len(f_lin_s_control), len(f_lin_s_fever)])

    pv_approx = [
        utilities.ks_bootstrap_test_by_fish(f_lin_s_control, f_lin_s_fever, 1000)[0],
        utilities.ks_bootstrap_test_by_fish(f_sq_s_control, f_sq_s_fever, 1000)[0]
    ]
    p_names = ["Linear approximation", "2nd order approximation"]
    approx_sig_df = generate_stat_df(np.array(pv_approx),np.vstack([fish_counts, fish_counts]),
                                  np.array(p_names))
    approx_sig_df.to_csv(path.join(analysis_dir, "stats_neuron_nonlinearity.csv"), index=False)

    d_test_corrs = pd.DataFrame(d_test_corrs)
    fig = lineplot(d_test_corrs, "Cumulative density", "Treatment", score_bc, "Test correlation", np.mean)
    pl.xlim(0.6, 1.0)
    sns.despine()
    fig.savefig(path.join(analysis_dir, "Test_correlation_comparison.pdf"))
    tc_sig_df = generate_stat_df(np.array([utilities.ks_bootstrap_test_by_fish(f_corr_control,
                                                                               f_corr_fever, 1000)[0]]),
                                 fish_counts[None, :],
                                 np.array(["Test correlation"]))
    tc_sig_df.to_csv(path.join(analysis_dir, "stats_fitneuron_testcorrs.csv"), index=False)

    # calculate delta-temperature
    aligned_1s_delta = compute_mm_delta_temps(temperature_interp, 5)


    # Predict transition probabilities, computed on the imaging traces, from the cluster activity as well
    # as the temperature stimulus itself as a control - since that stimulus "created" the activity of the response types
    # load stimulus driven model
    pred_valid = (temperature_interp >= 20).ravel()  # our stan models are only trained on data from 20-30C
    model_defs = {"PBS Control": {}, "poly(I:C)": {}}

    with h5py.File("E:\\Behavioral_Fever_Experiments\\gradient20-30_pre_proc\\plots\\fit.h5", 'r') as dfile:
        control_group = dfile["nac_PBS"]["Transition"]
        fever_group = dfile["nac_POLYIC"]["Transition"]
        model_defs["PBS Control"]["mm_icept"] = control_group["mm_icept"][()]
        model_defs["PBS Control"]["mm_orig"] = control_group["mm_orig"][()]
        model_defs["poly(I:C)"]["mm_icept"] = fever_group["mm_icept"][()]
        model_defs["poly(I:C)"]["mm_orig"] = fever_group["mm_orig"][()]

    # Compute transition probabilities across treatments
    n_samples = 100
    trans_probs = {k: np.zeros((temperature_interp.size, 3, 3)) for k in model_defs}
    trans_mats = {k: [] for k in model_defs}

    for k in model_defs:
        n_draws = model_defs[k]["mm_icept"].shape[0]
        for i, (temp, dtemp) in enumerate(zip(temperature_interp, aligned_1s_delta)):
            for s in range(n_samples):
                draw = np.random.randint(n_draws)
                trans_mat = tm_from_glm(temp-25, dtemp, model_defs[k]["mm_orig"][draw], model_defs[k]["mm_icept"][draw])
                trans_mats[k].append(trans_mat)
                tm = np.array(trans_mat)
                for j in range(3):
                    for jj in range(3):
                        trans_probs[k][i, j, jj] += tm[j, jj]
        trans_probs[k] /= n_samples

        trans_probs[k] = gaussian_filter1d(trans_probs[k], 5, axis=0)

    # Compute steady state probabilities
    steady_state = {k: np.zeros((temperature_interp.size, 3)) for k in model_defs}
    for k in model_defs:
        for i in range(temperature_interp.size):
            ss_probs = np.linalg.matrix_power(trans_probs[k][i], 100)
            for j in range(3):
                steady_state[k][i, j] = ss_probs[0, j]

    # plot steady state probabilities for treatments
    for i, name in enumerate(["p(Reversal)", "p(General)", "p(Persistent)"]):
        fig = pl.figure()
        for k in steady_state:
            pl.plot(neuron_time.ravel()[pred_valid], steady_state[k][pred_valid, i], '.', label=k)
        pl.legend()
        sns.despine()
        pl.title(name)
        pl.xlabel("Valid stimulus times")
        pl.ylabel("Steady state probability")
        fig.savefig(path.join(analysis_dir, "steady_state_" + name + ".pdf"))
