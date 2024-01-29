import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Some plotting config
sns.set(
    context="notebook",
    style="darkgrid",
    font="Times New Roman",
    font_scale=1.75,
)
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["mathtext.fontset"] = "cm"

import numpy as np
import scipy as sp
import pandas as pd

import os

os.environ["PYBASEBALL_CACHE"] = "/home/peter/projects/statcast-dashboard/.cache/pybaseball"

import pybaseball as pyb

pyb.cache.enable()

import streamlit as st

from astropy.stats import knuth_bin_width


# functions to extract median and 90th percentile launch speed for all players
def get_median_launch_speeds(bbes):
    # avoid looping over all players, pandas groupby is much faster
    return bbes["launch_speed"].groupby(bbes["batter_name"]).apply(np.nanmedian, include_groups=False)


def get_90th_launch_speeds(bbes):
    return (
        bbes["launch_speed"].groupby(bbes["batter_name"]).apply(lambda x: np.nanpercentile(x, 90), include_groups=False)
    )


def get_median_launch_angles(bbes):
    return bbes["launch_angle"].groupby(bbes["batter_name"]).apply(np.nanmedian, include_groups=False)


def get_optimal_pcts(bbes):
    # percentage of batted balls in the optimal range, per player

    # first, group by player
    by_player = bbes.groupby("batter_name")

    # next, compute the number of batted balls in the optimal range
    optimal = by_player.apply(lambda x: len(x[(x.launch_angle >= 15) & (x.launch_angle <= 40)]), include_groups=False)

    # compute the total number of batted balls
    total = by_player.apply(lambda x: len(x), include_groups=False)

    # compute the percentage
    return optimal / total * 100


def get_hardhit_pcts(bbes):
    # percentage of batted balls hit >= 95 mph, per player

    # first, group by player
    by_player = bbes.groupby("batter_name")

    # next, compute the number of batted balls hit >= 95 mph
    hardhit = by_player.apply(lambda x: len(x[x.launch_speed >= 95]), include_groups=False)

    # compute the total number of batted balls
    total = by_player.apply(lambda x: len(x), include_groups=False)

    # compute the percentage
    return hardhit / total * 100


def get_barrel_pcts(bbes):
    # percentage of batted balls hit >= 95 mph, in the optimal range, per player

    # first, group by player
    by_player = bbes.groupby("batter_name")

    # next, compute the number of batted balls hit >= 95 mph, in the optimal range
    barrel = by_player.apply(
        lambda x: len(x[(x.launch_speed >= 95) & (x.launch_angle >= 15) & (x.launch_angle <= 40)]), include_groups=False
    )

    # compute the total number of batted balls
    total = by_player.apply(lambda x: len(x), include_groups=False)

    # compute the percentage
    return barrel / total * 100


@st.cache_data
def get_data():
    bbes = pd.read_parquet("./bbes.parquet")

    # drop any players with fewer than 50 batted balls
    bbes = bbes.groupby("batter_name").filter(lambda x: len(x) >= 50)

    # sort by batter_name
    bbes = bbes.sort_values("batter_name")

    median_launch_speeds = get_median_launch_speeds(bbes).to_numpy()
    _90th_launch_speeds = get_90th_launch_speeds(bbes).to_numpy()
    median_launch_angles = get_median_launch_angles(bbes).to_numpy()
    optimal_angles = get_optimal_pcts(bbes).to_numpy()
    hard_hit_pcts = get_hardhit_pcts(bbes).to_numpy()
    barrel_pcts = get_barrel_pcts(bbes).to_numpy()

    # create a dataframe of these values, with player name as index
    league_values = pd.DataFrame(
        {
            "hitter_name": bbes["batter_name"].unique(),
            "median_launch_speed": median_launch_speeds,
            "90th_launch_speed": _90th_launch_speeds,
            "median_launch_angle": median_launch_angles,
            "optimal_pct": optimal_angles,
            "hard_hit_pct": hard_hit_pcts,
            "barrel_pct": barrel_pcts,
            "BBEs": bbes["batter_name"].value_counts().to_numpy(),
        }
    )
    return bbes, league_values


def main():
    st.title("Statcast Dashboard")

    st.sidebar.title("Statcast Dashboard")

    # add a sidebar, option for individual player or leaderboard
    st.sidebar.header("View")
    option = st.sidebar.radio("Select an option", ["Individual Player", "Leaderboard"])

    # Load data

    # time how long it takes to load data
    from time import time

    start = time()
    bbes, league_values = get_data()
    end = time()
    st.write(f"Data loaded in {end - start:.2f} seconds")

    if option == "Individual Player":
        # add a selectbox, over all players
        player = st.selectbox("Select a player", bbes["batter_name"].unique())

        st.subheader(f"Player: {player}")

        # filter data
        hitter = bbes[bbes["batter_name"] == player]

        # First, some league-wide stats, along with comparison to the player, including which percentile they are in

        # we want to show a table, with exit velocity, median, 90th percentile, league value, and percentile

        # league values
        league_median = np.nanmedian(league_values["median_launch_speed"])
        league_90th = np.nanmedian(league_values["90th_launch_speed"])
        league_hard_hit_pct = np.nanmedian(league_values["hard_hit_pct"])
        league_barrel_pct = np.nanmedian(league_values["barrel_pct"])

        # player values
        player_median = np.nanmedian(hitter["launch_speed"])
        player_90th = np.nanpercentile(hitter["launch_speed"], 90)
        player_hard_hit_pct = len(hitter[hitter.launch_speed >= 95]) / len(hitter) * 100
        player_barrel_pct = (
            len(hitter[(hitter.launch_speed >= 95) & (hitter.launch_angle >= 15) & (hitter.launch_angle <= 40)])
            / len(hitter)
            * 100
        )

        # compute percentiles
        median_percentile = sp.stats.percentileofscore(league_values["median_launch_speed"], player_median)
        _90th_percentile = sp.stats.percentileofscore(league_values["90th_launch_speed"], player_90th)
        hardhit_percentile = sp.stats.percentileofscore(league_values["hard_hit_pct"], player_hard_hit_pct)
        barrel_percentile = sp.stats.percentileofscore(league_values["barrel_pct"], player_barrel_pct)

        # create a dataframe
        summary_stats = pd.DataFrame(
            {
                "Stat": ["Median", "90th", "Hard Hit %", "Barrel %"],
                "Player": [player_median, player_90th, player_hard_hit_pct, player_barrel_pct],
                "League": [league_median, league_90th, league_hard_hit_pct, league_barrel_pct],
                "Percentile": [median_percentile, _90th_percentile, hardhit_percentile, barrel_percentile],
            }
        )
        summary_stats["Percentile"] = summary_stats["Percentile"].apply(lambda x: f"{x:.1f}")
        # drop the index
        summary_stats = summary_stats.set_index("Stat")

        # Now, we want to show a histogram of launch speeds, with the player's median and 90th percentile highlighted
        st.header("Launch Speed Distribution")

        # create a histogram
        fig, ax = plt.subplots()

        bins, bin_edges = knuth_bin_width(hitter["launch_speed"], return_bins=True)

        sns.histplot(hitter["launch_speed"], ax=ax, stat="count", kde=True, bins=bin_edges)

        # add vertical lines for median and 90th percentile
        ax.axvline(player_median, ls="-", color="k", label=f"Median: {player_median:.1f}", lw=2)
        ax.axvline(player_90th, ls="-", color="r", label=f"90th: {player_90th:.1f}", lw=2)

        # add the league median and 90th percentile in dashed lines
        ax.axvline(league_median, ls="--", color="k")
        ax.axvline(league_90th, ls="--", color="r")

        ax.set_xlabel("Launch Speed (mph)")
        ax.set_ylabel("Count")
        ax.legend()

        st.pyplot(fig)

        st.subheader("Summary Stats")

        st.table(summary_stats)

        # Now, we want to show a histogram of launch angles, with the player's median highlighted
        st.header("Launch Angle Distribution")

        # create a histogram
        fig, ax = plt.subplots()

        bins, bin_edges = knuth_bin_width(hitter["launch_angle"], return_bins=True)

        sns.histplot(hitter["launch_angle"], ax=ax, stat="count", kde=True, bins=bin_edges)

        # what percentage of the player's batted balls are in the optimal range?
        optimal = hitter[(hitter.launch_angle >= 15) & (hitter.launch_angle <= 40)]
        optimal_pct = len(optimal) / len(hitter) * 100

        # shade the optimal range
        ax.axvspan(15, 40, color="k", alpha=0.2, label=f"Optimal Range: {optimal_pct:.1f}%")

        ax.set_xlabel("Launch Angle (degrees)")
        ax.set_ylabel("Count")
        ax.legend()

        st.pyplot(fig)

        st.subheader("Summary Stats")

        st.table(
            pd.DataFrame(
                {
                    "Stat": ["Median", "Optimal %"],
                    "Player": [np.nanmedian(hitter["launch_angle"]), optimal_pct],
                    "League": [
                        np.nanmedian(league_values["median_launch_angle"]),
                        np.nanmedian(league_values["optimal_pct"]),
                    ],
                    "Percentile": [
                        sp.stats.percentileofscore(
                            league_values["median_launch_angle"], np.nanmedian(hitter["launch_angle"])
                        ),
                        sp.stats.percentileofscore(league_values["optimal_pct"], optimal_pct),
                    ],
                }
            ).set_index("Stat")
        )

    elif option == "Leaderboard":
        # display the sortable league leaderboard
        st.header("League Leaderboard")

        # add filters for min bbes
        min_bbes = st.slider("Minimum BBEs", 50, 400, 100)

        # filter the league values
        filtered_league_values = league_values[league_values["BBEs"] >= min_bbes]

        # display the leaderboard
        st.dataframe(
            filtered_league_values.sort_values("median_launch_speed", ascending=False).set_index("hitter_name")
        )


if __name__ == "__main__":
    main()
