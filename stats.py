from scipy.stats import norm, beta
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import itertools


def get_conversions_without_landing_page(conversions):
    # removes the landing_page_by_session column from the dataframe
    cleaning = """
        SELECT optimization, experience, experiment, variant
        , SUM(user_count) AS user_count
        , SUM(converted_user_count) as converted_user_count 
        FROM conversions
        GROUP BY optimization, experience, experiment, variant
        ORDER BY 1,2,3,4
    """
    return duckdb.query(cleaning).to_df()

def get_conversions_with_landing_page(conversions):
    sql = """
        SELECT optimization, experience, experiment
        , landing_page_by_session
        , CAST(SUM(user_count) AS int) AS user_count
        , CAST(SUM(converted_user_count) AS int) AS converted_user_count
        , SUM(converted_user_count)/SUM(user_count) AS conversion_rate 
        FROM conversions 
        GROUP BY optimization, experience, experiment, landing_page_by_session
        ORDER BY 1,2,3,4
    """
    return duckdb.query(sql).to_df()

def landing_page_stats(conversions):
    sql = """
        SELECT landing_page_by_session
        , CAST(SUM(user_count) AS int) AS user_count
        , CAST(SUM(converted_user_count) AS int) AS converted_user_count
        , SUM(converted_user_count)/SUM(user_count) AS conversion_rate 
        FROM conversions 
        GROUP BY landing_page_by_session
        ORDER BY conversion_rate DESC
    """
    return duckdb.query(sql).to_df()

def get_experiment_conversion_rate(conversions):
    sql = """
        SELECT optimization, experience, experiment
        , SUM(converted_user_count) AS converted_user_count
        , SUM(user_count) AS user_count
        , SUM(converted_user_count)/SUM(user_count) AS conversion_rate
        FROM conversions
        GROUP BY optimization, experience, experiment
    """
    diction = {}
    df = duckdb.query(sql).to_df()
    return df

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


def bayesian_stats(conversions, alpha_prior=1, beta_prior=1, resolution=500):
    """
    Creates and returns a new dataframe bayes_df, using information from dataframe df, containing the variant's experiment, variant name,
    conversion rate, improvement, probability to beat baseline, and probability to be best

    Parameters:
    - df: dataframe, contains a view of all of the variants from the optimization experiments

    Returns:
    - bayes_df: dataframe, contains bayesian statistics like the variant's experiment, variant name, conversion rate, improvement
                probability to beat baseline, and probability to be best
    """
    conversions = get_conversions_without_landing_page(conversions)
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
    conversions = get_conversions_without_landing_page(conversions)
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

def frequentist_audience_comparison(conversions):
    exp_conv = get_experiment_conversion_rate(conversions)
    conversions = get_conversions_with_landing_page(conversions)

    sql = """
    with extended as (
        select
        *, converted_user_count / user_count as conversion_rate
        from conversions
    )
    SELECT c.optimization, c.experience, c.experiment, c.landing_page_by_session
    , c.conversion_rate AS page_conversion_rate
    , e.conversion_rate AS experiment_conversion_rate
    , (c.conversion_rate - e.conversion_rate) / e.conversion_rate as lift
    , (c.converted_user_count + e.converted_user_count) / (c.user_count + e.user_count) as pool
    , sqrt(pool * (1 - pool) * (1 / c.user_count + 1 / e.user_count)) as se
    , (c.conversion_rate - e.conversion_rate) / se as z
    from extended c
    join exp_conv e
    ON c.optimization=e.optimization
    AND c.experience=e.experience
    AND c.experiment=e.experiment
    ORDER BY 1,2,3,4
    """
    df = duckdb.query(sql).to_df()
    df["confidence"] = abs(1 - (2 * (1 - norm.cdf(df.z))))
    df.drop(columns=["pool", "se", "z"], inplace=True)
    return df

def bayesian_audience_comparison(conversions, alpha_prior = 1, beta_prior = 1, resolution = 500):
    exp_conv = get_experiment_conversion_rate(conversions)
    conversions = get_conversions_with_landing_page(conversions)

    rows = []
    exp_samples = pd.DataFrame()
    for i in range(len(exp_conv)):
        total_exp = exp_conv.iloc[i]
        exp_samples[total_exp.experiment] = beta.rvs(
                alpha_prior + total_exp.converted_user_count,
                beta_prior + total_exp.user_count - total_exp.converted_user_count,
                size=400 * resolution,
        )
    for index in range(len(conversions)):
        page_con = conversions.iloc[index]
        page_sample = beta.rvs(
                alpha_prior + page_con.converted_user_count,
                beta_prior + page_con.user_count - page_con.converted_user_count,
                size=400 * resolution,
        )
        compare_sample = list(exp_samples[page_con.experiment])

        page_best = 0
        for sample_index in range(len(page_sample)):
            if page_sample[sample_index] > compare_sample[sample_index]:
                page_best += 1

        page_best /= len(page_sample)
        rows.append(
            {
                    "optimization": page_con.optimization,
                    "experience": page_con.experience,
                    "experiment": page_con.experiment,
                    "landing_page": page_con.landing_page_by_session,
                    "conversion_rate": page_con.converted_user_count / page_con.user_count,
                    "sample_mean": page_sample.mean(),
                    "sample_stdev": page_sample.std(),
                    "sample_min": min(page_sample),
                    "sample_max": max(page_sample),
                    "prob_beats": page_best,
                    "prob_best": page_best,
            }
        )
    return pd.DataFrame(rows)
        