import os
import glob
import re
import json

import numpy as np
import pandas as pd

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import linregress
from sensaug.dataset.augmentations import LEGACY20_OPS

matplotlib.use("Agg")
sns.set_theme(style="ticks")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SEEN_PERTURBATIONS = list(LEGACY20_OPS.keys())

UNSEEN_PERTURBATIONS = [
    "motionblur",
    "zoomblur",
    "pixelate",
    "jpegcompression",
    "snow",
    "frost",
    "fog",
]

PERTURBATION_ORDER = (
    ["clean"] + SEEN_PERTURBATIONS + ["combination"] + UNSEEN_PERTURBATIONS
)

AGGREGATE_CATEGORY_ORDER = ["a_clean", "b_seen", "combination", "d_imgnetc"]

plt.rcParams["figure.figsize"] = (20, 4)


def histogram_eval_csv(
    work_dir, pattern=None, dataset=None, backbone=None, name="temp", metric="mIoU"
):
    workdir_pattern = (
        os.path.join(work_dir, pattern)
        if pattern is not None
        else os.path.join(work_dir, "*")
    )
    all_workdirs = glob.glob(workdir_pattern)

    v1 = []  # imagenet-C
    v2 = []  # proxy
    v3 = []  # clean
    v4 = []  # BP
    v5 = []  # comb
    datasets = []
    backbones = []
    methods = []

    for w in sorted(all_workdirs):
        backbone_filter = True if backbone is None else backbone in w
        dataset_filter = True if dataset is None else dataset in w
        backbone_filter = backbone_filter and "convnext" not in w
        # dataset_filter = dataset_filter and "loveda" not in w

        exp_name = os.path.basename(w)
        info = exp_name.split("_")
        method_name, backbone_name, dataset_name = info[0], info[1], info[2]

        results_csv = os.path.join(w, "eval_results.csv")
        if os.path.exists(results_csv) and backbone_filter and dataset_filter:
            df = _get_pearson_sample(results_csv)

            print(df.head())
            imagenetc_miou = df[df.index == "dunseen"][metric]
            proxy_miou = df[df.index == "eproxy"][metric]
            clean_miou = df[df.index == "aclean"][metric]
            bp_miou = df[df.index == "bseen"][metric]
            comb_miou = df[df.index == "combination"][metric]

            if (
                len(imagenetc_miou) == 1
                and len(proxy_miou) == 1
                and len(clean_miou) == 1
            ):
                datasets.append(dataset_name)
                methods.append(method_name)
                backbones.append(backbone_name)
                v1.append(imagenetc_miou.item())
                v2.append(proxy_miou.item())
                v3.append(clean_miou.item())
                v4.append(bp_miou.item())
                v5.append(comb_miou.item())

    plot_data = pd.DataFrame(
        {
            "imgnetc": v1,
            "proxy": v2,
            "clean": v3,
            "BA": v4,
            "comb": v5,
            # 'dataset' : datasets,
            # 'backbone' : backbones,
            "method": ["random" if m == "default" else m for m in methods],
        }
    )

    melt_data = plot_data.melt(id_vars="method", value_name="miou", var_name="scenario")
    with sns.axes_style("darkgrid"):
        sns.barplot(
            data=melt_data,
            x="method",
            y="miou",
            hue="scenario",
            order=[
                "none",
                "random",
                "augmix",
                "autoaugment",
                "randaugment",
                "trivialaugment",
            ],
        )

    # dataset = "All" if dataset is None else dataset
    # dataset = "All" if dataset is None else dataset
    plt.title(f"Benchmark Performance Across Segmentation for {dataset.capitalize()}")
    plt.savefig(f"figs/benchmark_{name}.pdf", bbox_inches="tight", dpi=300)
    print(f"figs/benchmark_{name}.pdf")
    plt.close()


def _get_pearson_sample(csv_path):
    results_df = pd.read_csv(csv_path)
    # print(results_df.head())

    # add new column to parse perturbation name
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: x.replace("ptype=", "").replace("level=", "").replace("_", " ")
        if "ptype" in x or x == "clean"
        else "proxy"
    )
    results_df["level"] = results_df["perturbation"].apply(
        lambda x: (x.split(" ")[-1] if len(x.split(" ")) > 1 else None)
        if x != "proxy"
        else 3
    )
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: ("".join(x.split(" ")[:-1]) if len(x.split(" ")) > 1 else "clean")
        if x != "proxy"
        else "proxy"
    )

    # results_df = results_df.drop(results_df[(results_df.level == "5") & (results_df.perturbation.isin(SEEN_PERTURBATIONS))].index)

    aggregate_df = results_df.groupby(["perturbation"]).mean(numeric_only=True)
    aggregate_df["ptype"] = aggregate_df.index.to_series().apply(
        lambda x: "aclean"
        if x == "clean"
        else "bseen"
        if x in SEEN_PERTURBATIONS
        else "dunseen"
        if x in UNSEEN_PERTURBATIONS
        else "eproxy"
        if x == "proxy"
        else "combination"
    )

    # reindex aggregate df
    order = [x for x in PERTURBATION_ORDER if x in aggregate_df.index]
    aggregate_df = aggregate_df.reindex(order)
    aggregate_df = aggregate_df.drop(columns=["data_time", "time"])

    # print(aggregate_df)

    # aggregate_df now contains aggregate stats based on perturbation type
    aggregate2_df = aggregate_df.groupby(["ptype"]).mean(numeric_only=True)
    order2 = [x for x in AGGREGATE_CATEGORY_ORDER if x in aggregate2_df.index]
    aggregate2_df = aggregate2_df.reindex(order2)

    # print(aggregate2_df.head())

    # flattened_df = aggregate2_df.unstack().to_frame().sort_index(level=1).T
    # flattened_df.columns = flattened_df.columns.map('_'.join)

    # latex_string = flattened_df.style.format('{:.3f}').to_latex() #float_format="%.3f"
    # column_string = latex_string.split("\n")[-4].replace("\\\\", "").strip()[5:]
    # latex_string = latex_string.split("\n")[-3][3:] # get rid of a leading 0
    # latex_string = re.sub(' +',' ', latex_string.strip()).split("&")
    # column_string = re.sub(' +',' ', column_string.strip())

    # formatted_latex_string = []
    # for i, s in enumerate(latex_string):
    #     if i > 0 and i % 3 == 0:
    #         formatted_latex_string.append("\\phantom{ab}")
    #     formatted_latex_string.append(s)

    # formatted_latex_string = " & ".join(formatted_latex_string)

    print(os.path.basename(os.path.dirname(csv_path)))
    print(csv_path)
    # print(column_string)
    # print(formatted_latex_string)
    # print()

    # return aggregate2_df.iloc["dunseen"]['mIoU'], aggregate2_df.iloc["eproxy"]['mIoU']
    return aggregate2_df


def pearson_test_eccv(
    workdir, pattern=None, dataset=None, backbone=None, name="temp", metric="mIoU"
):
    workdir_pattern = (
        os.path.join(work_dir, pattern)
        if pattern is not None
        else os.path.join(workdir, "*")
    )
    all_workdirs = glob.glob(workdir_pattern)

    v1 = []  # imagenet-C
    v2 = []  # proxy
    v3 = []  # clean
    datasets = []
    backbones = []
    methods = []

    for w in sorted(all_workdirs):
        backbone_filter = True if backbone is None else backbone in w
        dataset_filter = True if dataset is None else dataset in w
        backbone_filter = backbone_filter and "convnext" not in w
        # dataset_filter = dataset_filter and "loveda" not in w

        exp_name = os.path.basename(w)
        info = exp_name.split("_")
        method_name, backbone_name, dataset_name = info[0], info[1], info[2]

        results_csv = os.path.join(w, "eval_results.csv")
        if os.path.exists(results_csv) and backbone_filter and dataset_filter:
            df = _get_pearson_sample(results_csv)

            print(df.head())
            imagenetc_miou = df[df.index == "dunseen"][metric]
            proxy_miou = df[df.index == "eproxy"][metric]
            clean_miou = df[df.index == "aclean"][metric]

            if (
                len(imagenetc_miou) == 1
                and len(proxy_miou) == 1
                and len(clean_miou) == 1
            ):
                datasets.append(dataset_name)
                methods.append(method_name)
                backbones.append(backbone_name)
                v1.append(imagenetc_miou.item())
                v2.append(proxy_miou.item())
                v3.append(clean_miou.item())

    plot_data = pd.DataFrame(
        {
            "imgnetc_miou": v1,
            "proxy_miou": v2,
            "clean_miou": v3,
            "dataset": datasets,
            "backbone": backbones,
            "method": methods,
        }
    )

    # print(v1, v2)
    print("pearson coefficient for imagenet-C vs. proxy")
    res = stats.pearsonr(v1, v2)
    r, p = res
    print(res)

    x, y = v1, v2
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    predicted_y = [(slope * xi) + intercept for xi in x]
    plot_data["yhat_proxy"] = predicted_y
    plot_data["resid_proxy"] = [
        observed_y - predicted_yi for observed_y, predicted_yi in zip(y, predicted_y)
    ]
    plot_data["resid_proxy"] = plot_data["resid_proxy"] / (
        max(plot_data["proxy_miou"]) - min(plot_data["proxy_miou"])
    )

    # result = smf.ols('proxy_miou ~ imgnetc_miou', plot_data).fit()
    # plot_data['yhat_proxy'] = result.fittedvalues
    # plot_data['resid_proxy'] = result.resid # / max(plot_data["proxy_miou"])
    # print(result.resid.abs().mean())

    print("pearson coefficient for imagenet-C vs. clean")
    res = stats.pearsonr(v1, v3)
    rc, pc = res
    print(res)

    x, y = v1, v3
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    predicted_y = [(slope * xi) + intercept for xi in x]
    plot_data["yhat_clean"] = predicted_y
    plot_data["resid_clean"] = [
        observed_y - predicted_yi for observed_y, predicted_yi in zip(y, predicted_y)
    ]
    plot_data["resid_clean"] = plot_data["resid_clean"] / (
        max(plot_data["clean_miou"]) - min(plot_data["clean_miou"])
    )

    # result = smf.ols('clean_miou ~ imgnetc_miou', plot_data).fit()
    # plot_data['yhat_clean'] = result.fittedvalues
    # plot_data['resid_clean'] = result.resid # / max(plot_data["clean_miou"])
    # print(result.resid.abs().mean())

    # x = list(set(backbones))
    # backbones_int = dict(zip(x, list(range(1,len(x)+1))))
    # colors = [backbones_int[v] for v in backbones]
    with sns.axes_style("darkgrid"):
        sns.scatterplot(
            data=plot_data, x="imgnetc_miou", y="proxy_miou", hue="backbone"
        )
        plt.plot(
            plot_data["imgnetc_miou"],
            plot_data["yhat_proxy"],
            color="red",
            label="Regression Line",
        )

    plt.title("Correlation Between Benchmark and Proxy Performance")
    plt.xlabel(f"ImageNet-C {metric.capitalize()}")
    plt.ylabel(f"Proxy {metric.capitalize()}")

    ax = plt.gca()  # Get a matplotlib's axes instance\
    # ax.scatter(v2, v1,c=colors, marker="o", cmap="bwr_r")

    plt.text(0.05, 0.8, "Pearson's r = {:.2f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.7, "p = {:.2E}".format(p), transform=ax.transAxes)
    plt.legend(loc="lower right")
    plt.savefig(f"figs/pearson_imgc_proxy_{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    with sns.axes_style("darkgrid"):
        sns.scatterplot(
            data=plot_data, x="imgnetc_miou", y="resid_proxy", hue="backbone"
        )

    plt.title("Residual Between Benchmark and Proxy Performance")
    plt.xlabel(f"ImageNet-C {metric.capitalize()}")
    plt.ylabel(f"Proxy {metric.capitalize()} Residual")

    ax = plt.gca()  # Get a matplotlib's axes instance\
    # ax.scatter(v2, v1,c=colors, marker="o", cmap="bwr_r")

    plt.ylim(-0.2, 0.2)
    plt.text(0.05, 0.8, "Pearson's r = {:.2f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.7, "p = {:.2E}".format(p), transform=ax.transAxes)
    plt.legend(loc="lower right")
    plt.savefig(f"figs/residual_imgc_proxy_{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # clean
    with sns.axes_style("darkgrid"):
        sns.scatterplot(
            data=plot_data, x="imgnetc_miou", y="clean_miou", hue="backbone"
        )
        plt.plot(
            plot_data["imgnetc_miou"],
            plot_data["yhat_clean"],
            color="red",
            label="Regression Line",
        )

    plt.title("Correlation Between Benchmark and Clean Performance")
    plt.xlabel(f"ImageNet-C {metric.capitalize()}")
    plt.ylabel(f"Clean {metric.capitalize()}")

    ax = plt.gca()  # Get a matplotlib's axes instance
    # ax.scatter(v3, v1, c=colors, marker="o", cmap="bwr_r")
    plt.text(0.05, 0.8, "Pearson's r = {:.2f}".format(rc), transform=ax.transAxes)
    plt.text(0.05, 0.7, "p = {:.2E}".format(pc), transform=ax.transAxes)
    plt.legend(loc="lower right")
    plt.savefig(f"figs/pearson_imgc_clean_{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    with sns.axes_style("darkgrid"):
        sns.scatterplot(
            data=plot_data, x="imgnetc_miou", y="resid_clean", hue="backbone"
        )

    plt.title("Residual Between Benchmark and Clean Performance")
    plt.xlabel(f"ImageNet-C {metric.capitalize()}")
    plt.ylabel(f"Clean {metric.capitalize()} Residual")

    ax = plt.gca()  # Get a matplotlib's axes instance\
    # ax.scatter(v2, v1,c=colors, marker="o", cmap="bwr_r")

    plt.ylim(-0.2, 0.2)
    plt.text(0.05, 0.8, "Pearson's r = {:.2f}".format(rc), transform=ax.transAxes)
    plt.text(0.05, 0.7, "p = {:.2E}".format(pc), transform=ax.transAxes)
    plt.legend(loc="lower right")
    plt.savefig(f"figs/residual_imgc_clean_{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # print("figs/pearson_imgc_clean_cityscapes.pdf")


def generate_latex_rows_from_workdir(workdir: str, pattern: str = None):
    workdir_pattern = (
        os.path.join(workdir, pattern)
        if pattern is not None
        else os.path.join(workdir, "*")
    )
    all_workdirs = glob.glob(workdir_pattern)

    for w in sorted(all_workdirs):
        results_csv = os.path.join(w, "eval_results.csv")
        if os.path.exists(results_csv):
            generate_latex_table_contents_from_csv(results_csv)


def generate_latex_table_contents_from_csv(csv_path: str):
    """Generates the contents of a latex table from CSV contents."""

    results_df = pd.read_csv(csv_path)
    # print(results_df.head())

    # add new column to parse perturbation name
    results_df = results_df[results_df["perturbation"] != "acdc"]
    results_df = results_df[results_df["perturbation"] != "idd"]
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: x.replace("ptype=", "").replace("level=", "").replace("_", " ")
    )
    results_df["level"] = results_df["perturbation"].apply(
        lambda x: x.split(" ")[-1] if len(x.split(" ")) > 1 else None
    )
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: "".join(x.split(" ")[:-1]) if len(x.split(" ")) > 1 else "clean"
    )

    # results_df = results_df.drop(results_df[(results_df.level == "5") & (results_df.perturbation.isin(SEEN_PERTURBATIONS))].index)

    aggregate_df = results_df.groupby(["perturbation"]).mean(numeric_only=True)
    aggregate_df["ptype"] = aggregate_df.index.to_series().apply(
        lambda x: "a_clean"
        if x == "clean"
        else "b_seen"
        if x in SEEN_PERTURBATIONS
        else "d_imgnetc"
        if x in UNSEEN_PERTURBATIONS
        else "combination"
    )

    # reindex aggregate df
    order = [x for x in PERTURBATION_ORDER if x in aggregate_df.index]
    aggregate_df = aggregate_df.reindex(order)
    aggregate_df = aggregate_df.drop(columns=["data_time", "time"])

    # print(aggregate_df)

    # aggregate_df now contains aggregate stats based on perturbation type
    aggregate2_df = aggregate_df.groupby(["ptype"]).mean(numeric_only=True)
    order2 = [x for x in AGGREGATE_CATEGORY_ORDER if x in aggregate2_df.index]
    aggregate2_df = aggregate2_df.reindex(order2)

    # aggregate2_df.drop(columns=["mDice"])
    if "mDice" in aggregate2_df.columns:
        aggregate2_df = aggregate2_df.drop(columns="mDice")
        # print(aggregate2_df)

    flattened_df = aggregate2_df.unstack().to_frame().sort_index(level=1).T
    flattened_df.columns = flattened_df.columns.map("_".join)

    # print(flattened_df)

    latex_string = flattened_df.style.format("{:.3f}").to_latex()  # float_format="%.3f"
    column_string = latex_string.split("\n")[-4].replace("\\\\", "").strip()[5:]
    latex_string = latex_string.split("\n")[-3][3:]  # get rid of a leading 0
    latex_string = re.sub(" +", " ", latex_string.strip()).split("&")
    column_string = re.sub(" +", " ", column_string.strip())

    formatted_latex_string = []
    for i, s in enumerate(latex_string):
        if i > 0 and i % 3 == 0:
            formatted_latex_string.append("\\phantom{ab}")
        formatted_latex_string.append(s)

    formatted_latex_string = " & ".join(formatted_latex_string)

    print(os.path.basename(os.path.dirname(csv_path)))
    print(csv_path)
    print(column_string)
    print(formatted_latex_string)
    print()

    return flattened_df


def generate_latex_table_contents_from_csv_icml(csv_path: str):
    """Generates the contents of a latex table from CSV contents."""

    results_df = pd.read_csv(csv_path)
    # print(results_df.head())

    # add new column to parse perturbation name
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: x.replace("ptype=", "").replace("level=", "").replace("_", " ")
    )
    results_df["level"] = results_df["perturbation"].apply(
        lambda x: x.split(" ")[-1] if len(x.split(" ")) > 1 else None
    )
    results_df["perturbation"] = results_df["perturbation"].apply(
        lambda x: "".join(x.split(" ")[:-1]) if len(x.split(" ")) > 1 else "clean"
    )

    # results_df = results_df.drop(results_df[(results_df.level == "5") & (results_df.perturbation.isin(SEEN_PERTURBATIONS))].index)

    aggregate_df = results_df.groupby(["perturbation"]).mean(numeric_only=True)
    aggregate_df["ptype"] = aggregate_df.index.to_series().apply(
        lambda x: "aclean"
        if x == "clean"
        else "bseen"
        if x in SEEN_PERTURBATIONS
        else "dunseen"
        if x in UNSEEN_PERTURBATIONS
        else "combination"
    )

    # reindex aggregate df
    order = [x for x in PERTURBATION_ORDER if x in aggregate_df.index]
    aggregate_df = aggregate_df.reindex(order)
    aggregate_df = aggregate_df.drop(columns=["data_time", "time"])

    # print(aggregate_df)

    # aggregate_df now contains aggregate stats based on perturbation type
    aggregate2_df = aggregate_df.groupby(["ptype"]).mean(numeric_only=True)
    order2 = [x for x in AGGREGATE_CATEGORY_ORDER if x in aggregate2_df.index]
    aggregate2_df = aggregate2_df.reindex(order2)

    flattened_df = aggregate2_df.unstack().to_frame().sort_index(level=1).T
    flattened_df.columns = flattened_df.columns.map("_".join)

    latex_string = flattened_df.style.format("{:.3f}").to_latex()  # float_format="%.3f"
    column_string = latex_string.split("\n")[-4].replace("\\\\", "").strip()[5:]
    latex_string = latex_string.split("\n")[-3][3:]  # get rid of a leading 0
    latex_string = re.sub(" +", " ", latex_string.strip()).split("&")
    column_string = re.sub(" +", " ", column_string.strip())

    formatted_latex_string = []
    for i, s in enumerate(latex_string):
        if i > 0 and i % 3 == 0:
            formatted_latex_string.append("\\phantom{ab}")
        formatted_latex_string.append(s)

    formatted_latex_string = " & ".join(formatted_latex_string)

    print(os.path.basename(os.path.dirname(csv_path)))
    print(csv_path)
    print(column_string)
    print(formatted_latex_string)
    print()


def plot_scaling_laws_data_from_csv(workdir: str, pattern: str = None):
    _methods = [
        "ours",
        "none",
        "augmix",
        "autoaugment",
        "trivialaugment",
        "randaugment",
    ]
    _indices = [1000, 2000, 3000, 4000, 5000]
    missing_train = []
    missing_test = []

    fig = plt.figure(figsize=(10, 10))
    sns.set_style("darkgrid")

    for m in _methods:
        x_values = [i for i in _indices]
        clean_miou = []
        single_perturb_miou = []
        combination_miou = []
        imgnet_miou = []
        skip_method = False

        for i in _indices:
            workdir_pattern = os.path.join(workdir, f"{m}_*_indices={str(i)}")
            exp_dir = glob.glob(workdir_pattern)
            if len(exp_dir) == 0:
                # skip_method = True
                missing_train.append(f"method={m}_indices={i}")
                x_values.remove(i)
                continue

            exp_dir = exp_dir[0]
            results_csv_path = os.path.join(exp_dir, "eval_results.csv")
            if os.path.exists(results_csv_path) == False:
                # skip_method = True
                missing_test.append(f"method={m}_indices={i}")
                x_values.remove(i)
                continue

            aggregate_results_df = generate_latex_table_contents_from_csv(
                results_csv_path
            )

            clean_miou.append(aggregate_results_df.iloc[0]["mIoU_a_clean"])
            single_perturb_miou.append(aggregate_results_df.iloc[0]["mIoU_b_seen"])
            combination_miou.append(aggregate_results_df.iloc[0]["mIoU_combination"])
            imgnet_miou.append(aggregate_results_df.iloc[0]["mIoU_d_imgnetc"])

        if skip_method == False:
            plt.plot(x_values, imgnet_miou, "-o", label=m)

    plt.legend()
    plt.title("Effect of Dataset Size on mIoU Performance", fontsize=24)
    plt.xlabel("# Samples")
    plt.ylabel("mIoU")
    plt.yscale("log")
    plt.savefig("output/scaling_laws_temp.png", bbox_inches="tight")

    print("missing train: ", missing_train)
    print("missing test: ", missing_test)


def plot_scaling_laws_data_from_realworld(
    workdir: str, dataset="acdc", pattern: str = None
):
    _methods = [
        "ours",
        "none",
        "augmix",
        "autoaugment",
        "trivialaugment",
        "randaugment",
        "idbh",
    ]
    _indices = [1000, 2000, 3000, 4000, 5000]
    missing_train = []
    missing_test = []

    plt.tight_layout()
    fig = plt.figure(figsize=(12, 8))
    sns.set_style("darkgrid")

    for m in _methods:
        x_values = [i for i in _indices]
        miou = []
        skip_method = False

        for i in _indices:
            workdir_pattern = os.path.join(workdir, f"{m}_*_indices={str(i)}")
            exp_dir = glob.glob(workdir_pattern)
            if len(exp_dir) == 0:
                # skip_method = True
                missing_train.append(f"method={m}_indices={i}")
                x_values.remove(i)
                continue

            exp_dir = exp_dir[0]
            json_path = os.path.join(exp_dir, dataset, "*", "*.json")
            json_path = glob.glob(json_path)
            if len(json_path) == 0:
                # skip_method = True
                missing_test.append(f"method={m}_indices={i}")
                x_values.remove(i)
                continue

            json_path = json_path[0]

            with open(json_path) as f:
                txt = f.read()
                txt = txt.split("}")[0] + "}"
                print(txt)
                data = json.loads(txt)

            miou.append(data["mIoU"])

        if skip_method == False:
            m = "baseline" if m == "none" else m
            plt.plot(x_values, miou, "-o", label=m)

    plt.legend(fontsize=18)

    title = (
        "Effect of Dataset Size on mIoU Performance In Domain Shift"
        if dataset == "idd"
        else "Effect of Dataset Size on mIoU Performance In Adverse Weather"
    )
    plt.title(title, fontsize=20)
    plt.xlabel("# Samples", fontsize=18)
    plt.ylabel("mIoU", fontsize=18)
    plt.savefig(f"output/scaling_laws_{dataset}.pdf", bbox_inches="tight")

    print("missing train: ", missing_train)
    print("missing test: ", missing_test)


def eval_results_to_csv(eval_results_path: str, savepath=None):
    """Converts evaluation results to CSV.
    Expects evaluation results to be in the format

    work_dirs/
    ├── eval_results_path
    │   ├── eval_clean
    │   │   ├── 20231229_203800
    │   │   └── vis_data
    │   │       └── vis_image
    │   ├── eval_ptype=blur_level=1
    │   │   ├── 20231229_221935
    │   │   └── vis_data
    │   │       └── vis_image
    │   ├── eval_ptype=blur_level=2
    │   │   ├── 20231229_222114
    │   │   └── vis_data
    │   │       └── vis_image
    ...
    """

    if savepath is None:
        savepath = os.path.join(eval_results_path, "eval_results.csv")

    header_logged = False
    subdirs = glob.glob(
        os.path.join(eval_results_path, "*")
    )  # eval_results_path/eval_ptype=...
    perturbations = [
        os.path.basename(x).replace("eval_", "") for x in subdirs
    ]  # ptype=..._level=...

    with open(savepath, "w+") as f:
        for i, subdir in enumerate(subdirs):
            json_files = sorted(glob.glob(os.path.join(subdir, "*", "*.json")))

            if len(json_files) > 0:
                jsonf = json_files[-1]  # get the most recent evaluation
                with open(jsonf) as jf:
                    json_text = jf.read().strip().split("}")[0] + "}"
                    json_data = json.loads(json_text)

                if header_logged == False:  # log header
                    column_names = ["perturbation"] + [
                        str(x) for x in list(json_data.keys())
                    ]
                    f.write(",".join(column_names) + "\n")
                    header_logged = True

                values = [perturbations[i]] + [str(x) for x in list(json_data.values())]
                f.write(",".join(values) + "\n")

        f.write("\n")

    print(f"CSV written to {savepath}")


if __name__ == "__main__":
    # work_dir = "/fs/nexus-projects/robustness_datasets/detection_work_dir"
    # name = "detection"
    # metric = "pascal_voc/mAP"
    # dataset = None

    work_dir = "/fs/nexus-projects/robustness_datasets/laura_work_dir"
    dataset = "cityscapes"
    backbone = "deeplabv3plus"
    name = "seg_all" if dataset is None else f"seg_{dataset}_{backbone}"
    metric = "mIoU"

    # work_dir = "/fs/nexus-projects/robustness_datasets/classification_work_dir"
    # name = "classification"
    # metric = "accuracy/top1"
    # dataset = None

    # work_dir = "./work_dirs"
    # generate_latex_rows_from_workdir(work_dir, pattern="dataset_experiments/*_eval")

    # pearson_test_eccv(work_dir, pattern="eccv/*_eval", name=name, metric=metric, dataset=dataset)
    histogram_eval_csv(
        work_dir,
        pattern="eccv/*deeplabv3plus*cityscapes_eval",
        name=name,
        metric=metric,
        dataset=dataset,
    )

    dataset = "potsdam"
    name = "seg_all" if dataset is None else f"seg_{dataset}_{backbone}"
    histogram_eval_csv(
        work_dir,
        pattern="eccv/*deeplabv3plus*potsdam_eval",
        name=name,
        metric=metric,
        dataset=dataset,
    )

    dataset = "loveda"
    name = "seg_all" if dataset is None else f"seg_{dataset}_{backbone}"
    histogram_eval_csv(
        work_dir,
        pattern="eccv/*deeplabv3plus*loveda_eval",
        name=name,
        metric=metric,
        dataset=dataset,
    )
