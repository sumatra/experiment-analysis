from scipy.stats import norm, beta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import itertools


def graph_prob_curves(stats):
    domain = np.linspace(stats.sample_min.min(), stats.sample_max.max())
    for variant, rows in stats.groupby("variant"):
        row = rows.iloc[0]
        plt.plot(
            domain, norm.pdf(domain, row.sample_mean, row.sample_stdev), label=variant
        )
    header = stats.iloc[0]
    plt.title(f"{header.optimization} {header.experience} ({header.experiment})")
    plt.xlabel("Conversion Rate")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()


def bayesian_stats(conversions, alpha_prior=1.0, beta_prior=1.0, resolution=500):
    """
    Creates and returns a new dataframe bayes_df, using information from dataframe df, containing the variant's experiment, variant name,
    conversion rate, improvement, probability to beat baseline, and probability to be best

    Parameters:
    - df: dataframe, contains a view of all of the variants from the optimization experiments

    Returns:
    - bayes_df: dataframe, contains bayesian statistics like the variant's experiment, variant name, conversion rate, improvement
                probability to beat baseline, and probability to be best
    """
    rows = []
    for _, grp in conversions.groupby(["optimization", "experience", "experiment"]):
        variants = grp.set_index("variant")
        samples = pd.DataFrame()
        for variant, row in variants.iterrows():
            samples[variant] = beta.rvs(
                alpha_prior + row.converted_user_count,
                beta_prior + row.user_count - row.converted_user_count,
                size=400 * resolution,
            )
        prob_best = samples.idxmax(axis=1).value_counts(normalize=True)
        mean = samples.mean()
        stdev = samples.std()
        minn = samples.min()
        maxx = samples.max()
        for variant, versus in itertools.product(variants.index, variants.index):
            row = variants.loc[variant]
            rows.append(
                {
                    "optimization": row.optimization,
                    "experience": row.experience,
                    "experiment": row.experiment,
                    "variant": variant,
                    "versus": versus,
                    "conversion_rate": row.converted_user_count / row.user_count,
                    "sample_mean": mean[variant],
                    "sample_stdev": stdev[variant],
                    "sample_min": minn[variant],
                    "sample_max": maxx[variant],
                    "prob_beats": (samples[variant] > samples[versus]).mean(),
                    "prob_best": prob_best.get(variant, 0),
                }
            )
    return pd.DataFrame(rows)


def frequentist_stats(conversions):
    """
    Creates and returns a dataframe that contains the experiment, experience, the variant that has the higher conversion rate,
    the variant that has the lower conversion rate, conversion lift, and confidence of pairs of variants belonging to the same experience.

    Parameters:
    - df: dataframe, contains a view of all of the variants from the optimization experiments

    Returns:
    - metrics_df: dataframe, contains the frequentist statistics of pairs of variants belonging to the same experience
    """
    sql = """
    with extended as (
        select
        *, converted_user_count / user_count as conversion_rate
        from conversions
    )
    select
    l.optimization, l.experience, l.experiment
    , l.variant, r.variant as versus
    , (l.conversion_rate - r.conversion_rate) / r.conversion_rate as lift
    , (l.converted_user_count + r.converted_user_count) / (l.user_count + r.user_count) as pool
    , sqrt(pool * (1 - pool) * (1 / l.user_count + 1 / r.user_count)) as se
    , (l.conversion_rate - r.conversion_rate) / se as z
    from extended l
    join extended r
    on l.optimization=r.optimization
    and l.experience=r.experience
    and l.experiment=r.experiment
    order by 1,2,3,4
    """
    df = duckdb.query(sql).to_df()
    df["confidence"] = abs(1 - (2 * (1 - norm.cdf(df.z))))
    df.drop(columns=["pool", "se", "z"], inplace=True)
    return df
