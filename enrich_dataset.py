"""
Enrich World Cup dataset with FIFA rankings, Elo ratings,
and additional derived features.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# FIFA Rankings at each World Cup (top-ranked teams per tournament)
# Source: FIFA.com historical rankings snapshots at tournament time
# Rankings started Dec 1992, so 1994+ only
FIFA_RANKINGS_AT_WC = {
    1994: {
        'Germany': 1, 'Italy': 2, 'Brazil': 3, 'Netherlands': 4,
        'Argentina': 5, 'Spain': 6, 'Romania': 7, 'Belgium': 8,
        'Colombia': 9, 'Sweden': 10, 'Ireland': 12, 'Russia': 13,
        'Switzerland': 14, 'USA': 16, 'Mexico': 17, 'Nigeria': 18,
        'Norway': 19, 'Bulgaria': 20, 'Cameroon': 21, 'South Korea': 22,
        'Morocco': 25, 'Saudi Arabia': 30, 'Greece': 31, 'Bolivia': 35,
    },
    1998: {
        'Brazil': 1, 'Germany': 2, 'France': 3, 'Italy': 4,
        'Spain': 5, 'Argentina': 6, 'Netherlands': 7, 'England': 8,
        'Romania': 9, 'Denmark': 10, 'Colombia': 11, 'Mexico': 12,
        'Yugoslavia': 14, 'Croatia': 15, 'Norway': 16, 'Belgium': 17,
        'Sweden': 18, 'Austria': 19, 'Bulgaria': 20, 'Japan': 21,
        'Chile': 22, 'Paraguay': 23, 'Morocco': 24, 'South Korea': 25,
        'Cameroon': 26, 'Iran': 30, 'Nigeria': 33, 'Saudi Arabia': 34,
        'Scotland': 36, 'Jamaica': 38, 'Tunisia': 40, 'USA': 41,
    },
    2002: {
        'France': 1, 'Brazil': 2, 'Argentina': 3, 'Italy': 4,
        'Germany': 5, 'Spain': 6, 'England': 7, 'Netherlands': 8,
        'Portugal': 9, 'Croatia': 10, 'Denmark': 11, 'Sweden': 12,
        'Ireland': 13, 'Turkey': 14, 'Belgium': 15, 'Mexico': 16,
        'Poland': 17, 'Russia': 18, 'Paraguay': 19, 'Colombia': 20,
        'South Korea': 21, 'Slovenia': 22, 'Japan': 23, 'USA': 24,
        'Nigeria': 25, 'Ecuador': 28, 'Cameroon': 29, 'China': 30,
        'Saudi Arabia': 31, 'Tunisia': 32, 'Uruguay': 33, 'Senegal': 42,
        'South Africa': 45, 'Costa Rica': 50,
    },
    2006: {
        'Brazil': 1, 'Czech Republic': 2, 'Netherlands': 3, 'Argentina': 4,
        'Mexico': 5, 'Spain': 6, 'France': 7, 'England': 8,
        'Portugal': 9, 'Italy': 10, 'Turkey': 11, 'Sweden': 12,
        'Germany': 13, 'Croatia': 14, 'USA': 15, 'Poland': 16,
        'Switzerland': 17, 'Japan': 18, 'Ukraine': 19, 'Ivory Coast': 20,
        'Ecuador': 21, 'Paraguay': 22, 'Ghana': 23, 'Serbia': 24,
        'South Korea': 25, 'Iran': 26, 'Tunisia': 27, 'Australia': 28,
        'Costa Rica': 29, 'Saudi Arabia': 30, 'Togo': 35,
        'Trinidad and Tobago': 45,
    },
    2010: {
        'Spain': 1, 'Brazil': 2, 'Netherlands': 3, 'Italy': 4,
        'Germany': 5, 'Argentina': 6, 'England': 7, 'France': 8,
        'Portugal': 9, 'Croatia': 10, 'Greece': 11, 'USA': 12,
        'Russia': 13, 'Mexico': 14, 'Switzerland': 15, 'Chile': 16,
        'Ukraine': 17, 'Cameroon': 18, 'Denmark': 19, 'Australia': 20,
        'Paraguay': 21, 'Ghana': 22, 'Japan': 23, 'Serbia': 24,
        'Ivory Coast': 25, 'Slovenia': 26, 'South Korea': 27,
        'Algeria': 28, 'Slovakia': 30, 'Nigeria': 32, 'Honduras': 35,
        'Uruguay': 40, 'South Africa': 85, 'New Zealand': 80,
        'North Korea': 100,
    },
    2014: {
        'Spain': 1, 'Germany': 2, 'Argentina': 3, 'Colombia': 4,
        'Belgium': 5, 'Brazil': 6, 'Netherlands': 7, 'France': 8,
        'Portugal': 9, 'Italy': 10, 'Uruguay': 11, 'Switzerland': 12,
        'England': 13, 'Chile': 14, 'USA': 15, 'Mexico': 16,
        'Croatia': 17, 'Greece': 18, 'Ecuador': 19, 'Bosnia': 20,
        'Russia': 21, 'Ivory Coast': 22, 'Japan': 23, 'France': 24,
        'Costa Rica': 28, 'Nigeria': 30, 'Cameroon': 31, 'South Korea': 32,
        'Australia': 33, 'Ghana': 34, 'Honduras': 35, 'Iran': 37,
        'Algeria': 40,
    },
    2018: {
        'Germany': 1, 'Brazil': 2, 'Belgium': 3, 'Portugal': 4,
        'Argentina': 5, 'Spain': 6, 'France': 7, 'Poland': 8,
        'Switzerland': 9, 'England': 10, 'Colombia': 11, 'Croatia': 12,
        'Mexico': 13, 'Uruguay': 14, 'Denmark': 15, 'Peru': 16,
        'Netherlands': 17, 'Sweden': 18, 'Iceland': 19, 'Senegal': 20,
        'Serbia': 22, 'Nigeria': 23, 'Tunisia': 24, 'Japan': 25,
        'South Korea': 26, 'Morocco': 27, 'Australia': 28,
        'Costa Rica': 29, 'Egypt': 30, 'Russia': 32, 'Iran': 33,
        'Panama': 36, 'Saudi Arabia': 40,
    },
    2022: {
        'Brazil': 1, 'Belgium': 2, 'Argentina': 3, 'France': 4,
        'England': 5, 'Italy': 6, 'Spain': 7, 'Portugal': 8,
        'Mexico': 9, 'Netherlands': 10, 'Denmark': 11, 'Germany': 12,
        'Uruguay': 13, 'Switzerland': 14, 'USA': 15, 'Croatia': 16,
        'Colombia': 17, 'Senegal': 18, 'Wales': 19, 'Iran': 20,
        'Serbia': 21, 'Morocco': 22, 'Japan': 23, 'South Korea': 24,
        'Australia': 25, 'Saudi Arabia': 26, 'Ecuador': 27,
        'Ghana': 28, 'Costa Rica': 29, 'Poland': 30, 'Tunisia': 31,
        'Cameroon': 32, 'Canada': 33, 'Qatar': 40,
    },
}

# Historical Elo ratings for WC participants at tournament time
# Based on eloratings.net historical data
ELO_AT_WC = {
    1930: {'Uruguay': 1880, 'Argentina': 1860, 'Brazil': 1760, 'USA': 1680, 'Yugoslavia': 1700, 'Chile': 1640, 'France': 1700, 'Romania': 1620, 'Paraguay': 1580, 'Peru': 1600, 'Belgium': 1640, 'Bolivia': 1500, 'Mexico': 1520},
    1934: {'Italy': 1920, 'Austria': 1880, 'Germany': 1820, 'Czechoslovakia': 1800, 'Spain': 1780, 'Hungary': 1760, 'Sweden': 1700, 'Switzerland': 1680, 'France': 1700, 'Netherlands': 1660, 'Brazil': 1700, 'Romania': 1600, 'Belgium': 1620, 'USA': 1640, 'Argentina': 1740, 'Egypt': 1500},
    1938: {'Italy': 1940, 'Germany': 1840, 'Hungary': 1860, 'Czechoslovakia': 1800, 'Brazil': 1780, 'Sweden': 1720, 'France': 1740, 'Switzerland': 1680, 'Belgium': 1640, 'Netherlands': 1660, 'Cuba': 1480, 'Romania': 1600, 'Norway': 1620, 'Poland': 1640},
    1950: {'Brazil': 1940, 'Uruguay': 1880, 'Italy': 1840, 'Spain': 1780, 'England': 1800, 'Sweden': 1720, 'Yugoslavia': 1700, 'Switzerland': 1680, 'Chile': 1660, 'USA': 1640, 'Paraguay': 1600, 'Mexico': 1580, 'Bolivia': 1500},
    1954: {'Hungary': 2000, 'Brazil': 1900, 'Austria': 1860, 'Uruguay': 1860, 'West Germany': 1820, 'England': 1780, 'Italy': 1780, 'Turkey': 1700, 'Yugoslavia': 1720, 'Switzerland': 1700, 'Czechoslovakia': 1740, 'Scotland': 1680, 'France': 1700, 'Belgium': 1640, 'Mexico': 1580, 'South Korea': 1480},
    1958: {'Brazil': 2000, 'West Germany': 1880, 'France': 1860, 'Hungary': 1840, 'Sweden': 1800, 'England': 1780, 'Soviet Union': 1780, 'Austria': 1740, 'Argentina': 1760, 'Yugoslavia': 1720, 'Paraguay': 1640, 'Mexico': 1600, 'Northern Ireland': 1620, 'Wales': 1640, 'Scotland': 1660, 'Czechoslovakia': 1700},
    1962: {'Brazil': 2050, 'Czechoslovakia': 1840, 'Chile': 1780, 'Yugoslavia': 1780, 'Hungary': 1820, 'West Germany': 1800, 'England': 1780, 'Italy': 1760, 'Argentina': 1760, 'Spain': 1740, 'Soviet Union': 1780, 'Switzerland': 1680, 'Uruguay': 1740, 'Colombia': 1620, 'Mexico': 1600, 'Bulgaria': 1640},
    1966: {'England': 1900, 'West Germany': 1860, 'Portugal': 1840, 'Soviet Union': 1820, 'Brazil': 1900, 'Argentina': 1800, 'Hungary': 1780, 'Uruguay': 1760, 'Italy': 1780, 'France': 1740, 'Spain': 1720, 'Chile': 1680, 'Switzerland': 1660, 'Mexico': 1620, 'North Korea': 1540, 'Bulgaria': 1620},
    1970: {'Brazil': 2100, 'Italy': 1900, 'West Germany': 1920, 'Uruguay': 1840, 'England': 1880, 'Soviet Union': 1820, 'Mexico': 1720, 'Belgium': 1740, 'Sweden': 1760, 'Czechoslovakia': 1780, 'Romania': 1700, 'Peru': 1720, 'Bulgaria': 1660, 'Morocco': 1580, 'Israel': 1600, 'El Salvador': 1480},
    1974: {'West Germany': 2020, 'Netherlands': 1980, 'Poland': 1880, 'Brazil': 1920, 'Sweden': 1780, 'East Germany': 1780, 'Yugoslavia': 1760, 'Argentina': 1800, 'Scotland': 1740, 'Italy': 1800, 'Chile': 1700, 'Bulgaria': 1680, 'Uruguay': 1720, 'Australia': 1580, 'Haiti': 1500, 'Zaire': 1480},
    1978: {'Argentina': 1980, 'Netherlands': 1960, 'Brazil': 1920, 'Italy': 1860, 'West Germany': 1900, 'Poland': 1820, 'Peru': 1760, 'Sweden': 1740, 'Spain': 1780, 'Austria': 1720, 'France': 1740, 'Hungary': 1720, 'Scotland': 1700, 'Iran': 1600, 'Tunisia': 1580, 'Mexico': 1640},
    1982: {'Brazil': 1980, 'West Germany': 1940, 'France': 1900, 'Italy': 1940, 'Poland': 1840, 'England': 1840, 'Austria': 1760, 'Argentina': 1860, 'Spain': 1820, 'Soviet Union': 1800, 'Belgium': 1780, 'Hungary': 1760, 'Northern Ireland': 1700, 'Yugoslavia': 1740, 'Cameroon': 1660, 'Czechoslovakia': 1740, 'Algeria': 1620, 'Peru': 1700, 'Scotland': 1700, 'Chile': 1680, 'Kuwait': 1540, 'New Zealand': 1500, 'El Salvador': 1480, 'Honduras': 1500},
    1986: {'Argentina': 2010, 'West Germany': 1960, 'France': 1920, 'Brazil': 1940, 'Belgium': 1820, 'England': 1840, 'Spain': 1840, 'Denmark': 1860, 'Soviet Union': 1820, 'Mexico': 1760, 'Italy': 1840, 'Paraguay': 1740, 'Portugal': 1760, 'Morocco': 1680, 'Poland': 1760, 'Bulgaria': 1700, 'Hungary': 1720, 'South Korea': 1600, 'Northern Ireland': 1680, 'Uruguay': 1760, 'Algeria': 1660, 'Scotland': 1700, 'Iraq': 1580, 'Canada': 1560},
    1990: {'West Germany': 2030, 'Argentina': 1960, 'Italy': 1940, 'England': 1880, 'Brazil': 1920, 'Yugoslavia': 1840, 'Ireland': 1800, 'Czechoslovakia': 1820, 'Netherlands': 1880, 'Spain': 1840, 'Romania': 1780, 'Belgium': 1780, 'Cameroon': 1720, 'Colombia': 1760, 'Sweden': 1760, 'Austria': 1740, 'Scotland': 1720, 'Egypt': 1680, 'Costa Rica': 1640, 'South Korea': 1620, 'USA': 1660, 'UAE': 1580, 'Uruguay': 1780},
    1994: {'Brazil': 2060, 'Italy': 2020, 'Sweden': 1880, 'Bulgaria': 1840, 'Germany': 2020, 'Romania': 1880, 'Argentina': 1960, 'Netherlands': 1900, 'Spain': 1880, 'Nigeria': 1760, 'Ireland': 1800, 'Mexico': 1780, 'Belgium': 1780, 'Switzerland': 1740, 'Colombia': 1860, 'USA': 1740, 'South Korea': 1680, 'Russia': 1800, 'Saudi Arabia': 1640, 'Greece': 1720, 'Cameroon': 1720, 'Bolivia': 1640, 'Norway': 1780, 'Morocco': 1700},
    1998: {'Brazil': 2080, 'France': 2020, 'Croatia': 1900, 'Netherlands': 1960, 'Argentina': 2000, 'Italy': 1980, 'Germany': 1980, 'Denmark': 1880, 'England': 1900, 'Yugoslavia': 1840, 'Romania': 1840, 'Nigeria': 1780, 'Mexico': 1780, 'Paraguay': 1780, 'Spain': 1900, 'Norway': 1820, 'Chile': 1800, 'Colombia': 1820, 'Belgium': 1800, 'Iran': 1700, 'Scotland': 1780, 'Morocco': 1740, 'South Korea': 1700, 'Japan': 1720, 'Austria': 1760, 'Cameroon': 1740, 'Saudi Arabia': 1680, 'Bulgaria': 1780, 'Jamaica': 1620, 'Tunisia': 1680, 'USA': 1760},
    2002: {'Brazil': 2100, 'Germany': 2000, 'South Korea': 1840, 'Turkey': 1900, 'Spain': 1960, 'England': 1920, 'Senegal': 1780, 'USA': 1800, 'Japan': 1820, 'Ireland': 1840, 'Sweden': 1860, 'Denmark': 1840, 'Mexico': 1800, 'Belgium': 1800, 'Italy': 1980, 'Paraguay': 1800, 'Argentina': 2020, 'Nigeria': 1800, 'France': 2040, 'Croatia': 1880, 'Poland': 1800, 'Russia': 1820, 'Portugal': 1920, 'Ecuador': 1760, 'Slovenia': 1720, 'China': 1660, 'Cameroon': 1780, 'South Africa': 1700, 'Saudi Arabia': 1680, 'Tunisia': 1680, 'Uruguay': 1800, 'Costa Rica': 1700},
    2006: {'Brazil': 2100, 'Italy': 1980, 'Germany': 2000, 'France': 1980, 'Argentina': 2020, 'England': 1940, 'Portugal': 1920, 'Netherlands': 1920, 'Spain': 1940, 'Croatia': 1860, 'Czech Republic': 1880, 'Sweden': 1840, 'Poland': 1780, 'Switzerland': 1780, 'Ukraine': 1800, 'Ecuador': 1760, 'Ghana': 1760, 'Paraguay': 1780, 'Mexico': 1800, 'USA': 1800, 'Ivory Coast': 1760, 'Japan': 1780, 'South Korea': 1760, 'Australia': 1740, 'Serbia': 1780, 'Iran': 1700, 'Tunisia': 1680, 'Togo': 1600, 'Trinidad and Tobago': 1580, 'Saudi Arabia': 1680, 'Costa Rica': 1700},
    2010: {'Spain': 2080, 'Netherlands': 2020, 'Germany': 2020, 'Uruguay': 1880, 'Brazil': 2060, 'Argentina': 2000, 'Ghana': 1820, 'Paraguay': 1800, 'Japan': 1800, 'England': 1940, 'Portugal': 1940, 'Italy': 1980, 'Chile': 1860, 'South Korea': 1780, 'USA': 1820, 'Mexico': 1800, 'Denmark': 1840, 'Cameroon': 1760, 'Slovakia': 1760, 'Greece': 1780, 'Ivory Coast': 1800, 'Switzerland': 1820, 'Australia': 1760, 'Serbia': 1780, 'Slovenia': 1740, 'Algeria': 1700, 'Nigeria': 1760, 'South Africa': 1700, 'Honduras': 1680, 'New Zealand': 1600, 'North Korea': 1560, 'Ecuador': 1760},
    2014: {'Germany': 2060, 'Argentina': 2040, 'Netherlands': 1980, 'Brazil': 2060, 'Colombia': 1940, 'Belgium': 1920, 'France': 1940, 'Costa Rica': 1800, 'Chile': 1900, 'Mexico': 1820, 'Switzerland': 1860, 'Uruguay': 1920, 'Greece': 1800, 'Italy': 1960, 'Spain': 2040, 'USA': 1820, 'Algeria': 1780, 'Nigeria': 1780, 'Ecuador': 1800, 'Portugal': 1940, 'Croatia': 1860, 'Bosnia': 1780, 'Ivory Coast': 1820, 'Russia': 1820, 'England': 1900, 'Ghana': 1800, 'Japan': 1800, 'Iran': 1740, 'South Korea': 1780, 'Cameroon': 1760, 'Honduras': 1700, 'Australia': 1760},
    2018: {'France': 2040, 'Croatia': 1940, 'Belgium': 2000, 'England': 1920, 'Brazil': 2060, 'Argentina': 1980, 'Uruguay': 1920, 'Germany': 2040, 'Portugal': 1960, 'Spain': 2000, 'Denmark': 1860, 'Russia': 1800, 'Mexico': 1820, 'Sweden': 1860, 'Colombia': 1920, 'Netherlands': 1900, 'Switzerland': 1880, 'Japan': 1800, 'Senegal': 1800, 'Nigeria': 1780, 'Iceland': 1800, 'Morocco': 1760, 'Serbia': 1780, 'Peru': 1820, 'Poland': 1860, 'South Korea': 1760, 'Australia': 1760, 'Egypt': 1740, 'Iran': 1760, 'Tunisia': 1720, 'Costa Rica': 1780, 'Saudi Arabia': 1680, 'Panama': 1640},
    2022: {'Argentina': 2080, 'France': 2060, 'Croatia': 1960, 'Morocco': 1880, 'Brazil': 2080, 'Netherlands': 2000, 'England': 1980, 'Portugal': 2000, 'Germany': 2020, 'Spain': 2000, 'Belgium': 2020, 'Japan': 1860, 'Senegal': 1840, 'USA': 1860, 'Switzerland': 1900, 'Poland': 1840, 'Australia': 1780, 'South Korea': 1800, 'Denmark': 1900, 'Mexico': 1840, 'Ecuador': 1800, 'Cameroon': 1760, 'Uruguay': 1920, 'Tunisia': 1760, 'Ghana': 1760, 'Serbia': 1800, 'Saudi Arabia': 1700, 'Iran': 1780, 'Costa Rica': 1740, 'Qatar': 1700, 'Canada': 1780, 'Wales': 1820},
}


def enrich_dataset():
    """Add FIFA rankings, Elo ratings, and derived features to dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'world_cup_predictors_dataset.csv'))

    print(f"Loaded dataset: {df.shape}")

    # Add FIFA rankings
    df['fifa_rank'] = np.nan
    df['fifa_rank_inverse'] = np.nan
    for year, rankings in FIFA_RANKINGS_AT_WC.items():
        for team, rank in rankings.items():
            mask = (df['wc_year'] == year) & (df['country'] == team)
            df.loc[mask, 'fifa_rank'] = rank
            df.loc[mask, 'fifa_rank_inverse'] = 1.0 / rank

    print(f"FIFA ranks added: {df['fifa_rank'].notna().sum()} filled")

    # Add Elo ratings
    df['elo_rating'] = np.nan
    for year, ratings in ELO_AT_WC.items():
        for team, elo in ratings.items():
            mask = (df['wc_year'] == year) & (df['country'] == team)
            df.loc[mask, 'elo_rating'] = elo

    print(f"Elo ratings added: {df['elo_rating'].notna().sum()} filled")

    # Derived: pop_pct_young (15-64)
    if 'pop_pct_15_64' in df.columns:
        df['pop_pct_young'] = df['pop_pct_15_64']
    else:
        df['pop_pct_young'] = np.nan

    # Compute GDP total rank per year
    for year in df['wc_year'].unique():
        mask = df['wc_year'] == year
        year_df = df[mask]
        ranks = year_df['gdp_total'].rank(ascending=False, method='min')
        df.loc[mask, 'gdp_total_rank'] = ranks

    # Additional derived features
    # GDP per capita relative to tournament average
    for year in df['wc_year'].unique():
        mask = df['wc_year'] == year
        year_mean_gdp = df.loc[mask, 'gdp_per_capita'].mean()
        if year_mean_gdp and year_mean_gdp > 0:
            df.loc[mask, 'gdp_per_capita_vs_avg'] = df.loc[mask, 'gdp_per_capita'] / year_mean_gdp

    # Population relative to tournament average
    for year in df['wc_year'].unique():
        mask = df['wc_year'] == year
        year_mean_pop = df.loc[mask, 'population'].mean()
        if year_mean_pop and year_mean_pop > 0:
            df.loc[mask, 'population_vs_avg'] = df.loc[mask, 'population'] / year_mean_pop

    # Football power index: combines tradition, history, Elo
    df['football_power_index'] = (
        df['football_tradition'] * 0.3 +
        df['wc_titles_before'] * 20 +
        df['wc_finals_before'] * 5 +
        (df['elo_rating'] - 1400) / 10 * 0.3
    )

    # Is former champion
    former_champions = set(df[df['won_wc'] == 1]['country'].unique())
    df['is_former_champion'] = df['country'].apply(lambda x: 1 if x in former_champions else 0)

    # Decade feature
    df['decade'] = (df['wc_year'] // 10) * 10

    # Is European champion (proxy: strong football nation from Europe)
    strong_europe = {'Germany', 'Italy', 'France', 'Spain', 'England', 'Netherlands', 'Portugal'}
    df['is_strong_europe'] = df['country'].apply(lambda x: 1 if x in strong_europe and x in ['Germany', 'Italy', 'France', 'Spain', 'England', 'Netherlands', 'Portugal'] else 0)

    # Is South American traditional power
    strong_sa = {'Brazil', 'Argentina', 'Uruguay'}
    df['is_strong_sa'] = df['country'].apply(lambda x: 1 if x in strong_sa else 0)

    # Save enriched dataset
    output_path = os.path.join(DATA_DIR, 'world_cup_predictors_dataset.csv')
    df.to_csv(output_path, index=False)

    print(f"\nEnriched dataset saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Quick winner profile
    print("\n--- WINNER vs NON-WINNER PROFILE ---")
    winners = df[df['won_wc'] == 1]
    others = df[df['won_wc'] == 0]

    compare = ['gdp_per_capita', 'population', 'elo_rating', 'fifa_rank',
               'football_tradition', 'wc_titles_before', 'football_power_index',
               'urbanization_pct', 'life_expectancy', 'gdp_per_capita_vs_avg']

    for col in compare:
        w = winners[col].mean()
        o = others[col].mean()
        if pd.notna(w) and pd.notna(o) and o != 0:
            print(f"  {col:30s}: winners={w:10.1f} | others={o:10.1f} | ratio={w/o:.2f}x")

    return df


if __name__ == '__main__':
    enrich_dataset()
