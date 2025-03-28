import argparse
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

from aida_mnist.utils import calculate_sens_spec, update_positives_negatives, set_plot_style


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--s_des", type=str, default="all", choices=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "all"]
    )
    args = parser.parse_args()

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    reps = args.reps
    p_rand = 0.1
    u_normalization = True
    impute = True
    batch_size = 128
    update_every = 5
    log_window = 1000
    s_des_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if args.s_des == "all" else [float(args.s_des)]

    fig, axs = plt.subplots(2, 3, figsize=(6.50127, 0.7 * 6.50127))
    # create rainbow color map
    cmap = sns.color_palette("rainbow", 9)
    tp = np.zeros((9, reps))
    fp = np.zeros((9, reps))
    tn = np.zeros((9, reps))
    fn = np.zeros((9, reps))
    for sens_i, s_des in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        sens = np.zeros((reps, (60_000 // batch_size)))
        spec = np.zeros((reps, (60_000 // batch_size)))
        system_success = np.zeros((reps, (60_000 // batch_size)))
        novice_success = np.zeros((reps, (60_000 // batch_size)))
        query_rate = np.zeros((reps, (60_000 // batch_size)))
        train_samples = np.zeros((reps, (60_000 // batch_size)))

        for i in range(reps):
            dir_name = f"i{i}_r{p_rand}_s{s_des}_u{u_normalization}_i{impute}_b{batch_size}_e{update_every}"
            save_path = Path("results") / dir_name / f"{i}"
            if (save_path / "results.npy").exists():
                results = np.load(save_path / "results.npy", allow_pickle=True).item()
            else:
                print(("-" * 80) + f"\n{dir_name} not found\n" + ("-" * 80))
                continue
            c = np.asarray(results["c"])
            q = np.asarray(results["q"])
            f_window = []
            s_window = []
            t_samples = 0
            for j in range(sens.shape[1]):
                # Calculate sensitivity and specificity (including random queries)
                q_batch = q[j * batch_size : (j + 1) * batch_size]
                c_batch = c[j * batch_size : (j + 1) * batch_size]
                tp[sens_i, i], fp[sens_i, i], tn[sens_i, i], fn[sens_i, i] = update_positives_negatives(
                    tp[sens_i, i], fp[sens_i, i], tn[sens_i, i], fn[sens_i, i], q_batch, c_batch
                )
                sensitivity, specificity, f_window, s_window = calculate_sens_spec(
                    f_window, s_window, q_batch, c_batch, window=log_window
                )
                sens[i, j] = sensitivity
                spec[i, j] = specificity
                t_samples += np.sum(q_batch)
                train_samples[i, j] = t_samples
                q_batch = q[max((j + 1) * batch_size - log_window, 0) : (j + 1) * batch_size]
                c_batch = c[max((j + 1) * batch_size - log_window, 0) : (j + 1) * batch_size]
                window_len = len(c_batch)
                system_success[i, j] = np.sum(np.logical_or(c_batch, q_batch)) / window_len
                novice_success[i, j] = np.sum(c_batch) / window_len
                query_rate[i, j] = np.sum(q_batch) / window_len

        sens_mean = np.nanmean(sens, axis=0)
        sens_std = np.nanstd(sens, axis=0)

        spec_mean = np.nanmean(spec, axis=0)
        spec_std = np.nanstd(spec, axis=0)

        system_success_mean = np.nanmean(system_success, axis=0)
        system_success_std = np.nanstd(system_success, axis=0)

        novice_success_mean = np.nanmean(novice_success, axis=0)
        novice_success_std = np.nanstd(novice_success, axis=0)

        query_rate_mean = np.nanmean(query_rate, axis=0)
        query_rate_std = np.nanstd(query_rate, axis=0)

        train_samples_mean = np.nanmean(train_samples, axis=0)
        train_samples_std = np.nanstd(train_samples, axis=0)

        x = np.arange(sens.shape[1])

        axs[0, 0].plot(x, sens_mean, label=r"$\sigma_\mathrm{{des}}={}$".format(s_des), color=cmap[sens_i])
        axs[0, 0].fill_between(x, sens_mean - sens_std, sens_mean + sens_std, alpha=0.3, color=cmap[sens_i])
        axs[0, 0].plot(x, np.ones_like(x) * s_des, "--", color=cmap[sens_i])
        axs[0, 0].set_ylabel("Sensitivity")
        axs[0, 0].set_ylim(0, 1)

        axs[0, 1].plot(x, spec_mean, color=cmap[sens_i])
        axs[0, 1].fill_between(x, spec_mean - spec_std, spec_mean + spec_std, alpha=0.3, color=cmap[sens_i])
        axs[0, 1].set_ylabel("Specificity")
        axs[0, 1].set_ylim(0, 1)

        axs[0, 2].plot(x, query_rate_mean, color=cmap[sens_i])
        axs[0, 2].fill_between(
            x, query_rate_mean - query_rate_std, query_rate_mean + query_rate_std, alpha=0.3, color=cmap[sens_i]
        )
        axs[0, 2].set_ylabel("Query Rate")
        axs[0, 2].set_ylim(0, 1)

        axs[1, 0].plot(x, train_samples_mean, color=cmap[sens_i])
        axs[1, 0].fill_between(
            x, train_samples_mean - train_samples_std, train_samples_mean + train_samples_std, alpha=0.3, color=cmap[sens_i]
        )
        axs[1, 0].set_ylabel("Train Samples")

        axs[1, 1].plot(x, novice_success_mean, color=cmap[sens_i])
        axs[1, 1].fill_between(
            x, novice_success_mean - novice_success_std, novice_success_mean + novice_success_std, alpha=0.3, color=cmap[sens_i]
        )
        axs[1, 1].set_ylabel("Novice Success Rate")
        axs[1, 1].set_ylim(0, 1)

        axs[1, 2].plot(x, system_success_mean, color=cmap[sens_i])
        axs[1, 2].fill_between(
            x, system_success_mean - system_success_std, system_success_mean + system_success_std, alpha=0.3, color=cmap[sens_i]
        )
        axs[1, 2].set_ylabel("System Success Rate")
        axs[1, 2].set_ylim(0, 1)
    for ax in axs.flatten():
        ax.ticklabel_format(axis="y", style="sci", scilimits=(2, 1))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append(r"Desired $\bf{(A)}$")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}", r"\textbf{E}", r"\textbf{F}"]
    for ax, label in zip(axs.flatten(), labels):
        ax.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    axs[1, 0].set_xlabel("Step")
    axs[1, 1].set_xlabel("Step")
    axs[1, 2].set_xlabel("Step")
    fig.savefig("figures/mnist.pdf")

    # Create table
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    sens_mean = np.mean(sens, axis=1)
    sens_std = np.std(sens, axis=1)
    spec_mean = np.mean(spec, axis=1)
    spec_std = np.std(spec, axis=1)
    for i, s_des in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        print(
            f"{s_des} & {sens_mean[i]:.3f} $\pm$ {sens_std[i]:.3f} & {spec_mean[i]:.3f} $\pm$ {spec_std[i]:.3f} & {sens_mean[i]+spec_mean[i]:.3f} \\\\"
        )
