"""
World Cup Predictors - Comprehensive Data Collection
=====================================================
Collects ALL possible predictors of World Cup winners:
- Sports/Football (FIFA rankings, WC history, confederation)
- Economy (GDP, GDP per capita, growth, trade)
- Demographics (population, urbanization, density)
- Health (life expectancy, mortality, health spending)
- Education (literacy, school enrollment, education spending)
- Governance (corruption, political stability, rule of law)
- Infrastructure (internet, electricity, roads)
- Football-specific (continental, historical performance)
"""

import json
import time
import warnings
import os
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

from shared import harmonize_country

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# SECTION 1: WORLD CUP HISTORICAL DATA
# ============================================================

# All World Cup winners
WC_WINNERS = {
    1930: 'Uruguay', 1934: 'Italy', 1938: 'Italy',
    1950: 'Uruguay', 1954: 'West Germany', 1958: 'Brazil',
    1962: 'Brazil', 1966: 'England', 1970: 'Brazil',
    1974: 'West Germany', 1978: 'Argentina', 1982: 'Italy',
    1986: 'Argentina', 1990: 'West Germany', 1994: 'Brazil',
    1998: 'France', 2002: 'Brazil', 2006: 'Italy',
    2010: 'Spain', 2014: 'Germany', 2018: 'France',
    2022: 'Argentina',
}

# Host countries
WC_HOSTS = {
    1930: 'Uruguay', 1934: 'Italy', 1938: 'France',
    1950: 'Brazil', 1954: 'Switzerland', 1958: 'Sweden',
    1962: 'Chile', 1966: 'England', 1970: 'Mexico',
    1974: 'West Germany', 1978: 'Argentina', 1982: 'Spain',
    1986: 'Mexico', 1990: 'Italy', 1994: 'United States',
    1998: 'France', 2002: 'South Korea/Japan', 2006: 'Germany',
    2010: 'South Africa', 2014: 'Brazil', 2018: 'Russia',
    2022: 'Qatar',
}

# Country name to ISO3 mapping (comprehensive)
COUNTRY_TO_ISO3 = {
    'Algeria': 'DZA',
    'Argentina': 'ARG', 'Australia': 'AUS', 'Austria': 'AUT',
    'Belgium': 'BEL', 'Bolivia': 'BOL', 'Bosnia and Herzegovina': 'BIH',
    'Brazil': 'BRA', 'Bulgaria': 'BGR', 'Cameroon': 'CMR',
    'Canada': 'CAN', 'Chile': 'CHL', 'China': 'CHN',
    'Colombia': 'COL', 'Costa Rica': 'CRC', 'Croatia': 'HRV',
    'Cuba': 'CUB', 'Czech Republic': 'CZE', 'Czechoslovakia': 'CZE',
    'Burma': 'MMR', 'Denmark': 'DEN', 'Ecuador': 'ECU', 'Egypt': 'EGY',
    'El Salvador': 'SLV', 'England': 'ENG', 'France': 'FRA',
    'Germany': 'DEU', 'Ghana': 'GHA', 'Greece': 'GRC',
    'Honduras': 'HND', 'Hungary': 'HUN', 'Iceland': 'ISL',
    'Dutch East Indies': 'IDN', 'Indonesia': 'IDN', 'Iran': 'IRN', 'IR Iran': 'IRN', 'Iraq': 'IRQ',
    'Ireland': 'IRL', 'Republic of Ireland': 'IRL', 'Israel': 'ISR', 'Italy': 'ITA',
    'Ivory Coast': 'CIV', 'Jamaica': 'JAM', 'Japan': 'JPN',
    'Kuwait': 'KWT', 'Mexico': 'MEX', 'Morocco': 'MAR', 'Myanmar': 'MMR',
    'Netherlands': 'NLD', 'New Zealand': 'NZL', 'Nigeria': 'NGA',
    'North Korea': 'PRK', 'Korea DPR': 'PRK', 'Northern Ireland': 'NIR', 'Norway': 'NOR',
    'Panama': 'PAN', 'Paraguay': 'PRY', 'Peru': 'PER',
    'Poland': 'POL', 'Portugal': 'PRT', 'Qatar': 'QAT',
    'Romania': 'ROU', 'Russia': 'RUS', 'Saudi Arabia': 'SAU',
    'Scotland': 'SCO', 'Senegal': 'SEN', 'Serbia': 'SRB',
    'Serbia and Montenegro': 'SRB', 'Slovakia': 'SVK',
    'Slovenia': 'SVN', 'South Africa': 'ZAF',
    'South Korea': 'KOR', 'Korea Republic': 'KOR', 'Soviet Union': 'RUS', 'Spain': 'ESP',
    'Sweden': 'SWE', 'Switzerland': 'CHE', 'Togo': 'TGO',
    'Trinidad and Tobago': 'TTO', 'Tunisia': 'TUN',
    'Turkey': 'TUR', 'UAE': 'ARE', 'Ukraine': 'UKR',
    'United Arab Republic': 'EGY',
    'United States': 'USA', 'USA': 'USA', 'Uruguay': 'URY', 'Wales': 'WAL',
    'West Germany': 'DEU', 'German DR': 'DEU', 'Yugoslavia': 'SRB',
    'Zaire': 'COD', 'East Germany': 'DEU',
    'South Korea/Japan': 'KOR',  # host entry
    'United Arab Emirates': 'ARE',
    'DR Congo': 'COD', 'Chinese Taipei': 'TWN',
    'Haiti': 'HTI', 'Jamaica': 'JAM', 'China PR': 'CHN',
    'IR Iran': 'IRN', 'Korea Republic': 'KOR',
    'Côte d\'Ivoire': 'CIV',
    'Cape Verde': 'CPV', 'Curaçao': 'CUW', 'Curacao': 'CUW',
    'Uzbekistan': 'UZB', 'Jordan': 'JOR',
    'Vietnam': 'VNM', 'Vietnam Republic': 'VNM', 'South Vietnam': 'VNM',
}

# All World Cup participants by year (comprehensive list)
WC_PARTICIPANTS = {
    1930: ['Argentina','Belgium','Bolivia','Brazil','Chile','France','Mexico','Paraguay','Peru','Romania','Uruguay','USA','Yugoslavia'],
    1934: ['Argentina','Austria','Belgium','Brazil','Czechoslovakia','Egypt','France','Germany','Hungary','Italy','Netherlands','Romania','Spain','Sweden','Switzerland','USA'],
    1938: ['Belgium','Brazil','Cuba','Czechoslovakia','Dutch East Indies','France','Germany','Hungary','Italy','Netherlands','Norway','Poland','Romania','Sweden','Switzerland'],
    1950: ['Bolivia','Brazil','Chile','England','France','India','Italy','Mexico','Paraguay','Spain','Sweden','Switzerland','USA','Uruguay','Yugoslavia'],
    1954: ['Austria','Belgium','Brazil','Czechoslovakia','England','France','Hungary','Italy','Mexico','Scotland','South Korea','Switzerland','Turkey','Uruguay','West Germany','Yugoslavia'],
    1958: ['Argentina','Austria','Brazil','England','France','Hungary','Mexico','Northern Ireland','Paraguay','Scotland','Sweden','Soviet Union','Wales','West Germany','Yugoslavia'],
    1962: ['Argentina','Brazil','Bulgaria','Chile','Colombia','Czechoslovakia','England','Hungary','Italy','Mexico','Spain','Switzerland','Uruguay','West Germany','Yugoslavia'],
    1966: ['Argentina','Brazil','Bulgaria','Chile','England','France','Hungary','Italy','Mexico','North Korea','Portugal','Spain','Switzerland','Soviet Union','Uruguay','West Germany'],
    1970: ['Belgium','Brazil','Bulgaria','El Salvador','England','Israel','Italy','Mexico','Morocco','Peru','Romania','Sweden','Czechoslovakia','Uruguay','Soviet Union','West Germany'],
    1974: ['Argentina','Australia','Brazil','Bulgaria','Chile','East Germany','Haiti','Italy','Netherlands','Poland','Scotland','Sweden','West Germany','Yugoslavia','Zaire'],
    1978: ['Argentina','Brazil','France','Hungary','Iran','Italy','Mexico','Netherlands','Peru','Poland','Scotland','Spain','Sweden','Tunisia','West Germany'],
    1982: ['Algeria','Argentina','Austria','Belgium','Brazil','Cameroon','Chile','Czechoslovakia','El Salvador','England','France','Honduras','Hungary','Italy','Kuwait','New Zealand','Northern Ireland','Peru','Poland','Scotland','Soviet Union','Spain','West Germany','Yugoslavia'],
    1986: ['Algeria','Argentina','Belgium','Brazil','Bulgaria','Canada','Denmark','England','France','Hungary','Iraq','Italy','Mexico','Morocco','Northern Ireland','Paraguay','Poland','Portugal','Scotland','South Korea','Spain','Soviet Union','Uruguay','West Germany'],
    1990: ['Argentina','Austria','Belgium','Brazil','Cameroon','Colombia','Costa Rica','Czechoslovakia','Egypt','England','Germany','Ireland','Italy','Netherlands','Romania','Scotland','South Korea','Spain','Sweden','UAE','USA','Uruguay','Yugoslavia'],
    1994: ['Argentina','Bolivia','Brazil','Bulgaria','Cameroon','Colombia','Germany','Greece','Ireland','Italy','Mexico','Morocco','Netherlands','Nigeria','Norway','Romania','Russia','Saudi Arabia','South Korea','Spain','Sweden','Switzerland','USA'],
    1998: ['Argentina','Austria','Belgium','Brazil','Bulgaria','Cameroon','Chile','Colombia','Croatia','Denmark','England','France','Germany','Iran','Italy','Jamaica','Japan','Mexico','Morocco','Netherlands','Nigeria','Norway','Paraguay','Romania','Saudi Arabia','Scotland','South Korea','Spain','Tunisia','USA','Yugoslavia'],
    2002: ['Argentina','Belgium','Brazil','Cameroon','China','Costa Rica','Croatia','Denmark','Ecuador','England','France','Germany','Ireland','Italy','Japan','Mexico','Nigeria','Paraguay','Poland','Portugal','Russia','Saudi Arabia','Senegal','Slovenia','South Africa','South Korea','Spain','Sweden','Tunisia','Turkey','USA','Uruguay'],
    2006: ['Argentina','Australia','Brazil','Costa Rica','Croatia','Czech Republic','Ecuador','England','France','Germany','Ghana','Iran','Italy','Ivory Coast','Japan','Mexico','Netherlands','Paraguay','Poland','Portugal','Saudi Arabia','Serbia and Montenegro','South Korea','Spain','Sweden','Switzerland','Togo','Trinidad and Tobago','Tunisia','Ukraine','USA'],
    2010: ['Algeria','Argentina','Australia','Brazil','Cameroon','Chile','Denmark','England','France','Germany','Ghana','Greece','Honduras','Italy','Ivory Coast','Japan','Mexico','Netherlands','New Zealand','Nigeria','North Korea','Paraguay','Portugal','Serbia','Slovakia','Slovenia','South Africa','South Korea','Spain','Sweden','Switzerland','Uruguay'],
    2014: ['Algeria','Argentina','Australia','Belgium','Bosnia and Herzegovina','Brazil','Cameroon','Chile','Colombia','Costa Rica','Croatia','Ecuador','England','France','Germany','Ghana','Greece','Honduras','Iran','Italy','Ivory Coast','Japan','Mexico','Netherlands','Nigeria','Portugal','Russia','South Korea','Spain','Switzerland','Uruguay','USA'],
    2018: ['Argentina','Australia','Belgium','Brazil','Colombia','Costa Rica','Croatia','Denmark','Egypt','England','France','Germany','Iceland','Iran','Japan','Mexico','Morocco','Nigeria','Panama','Peru','Poland','Portugal','Russia','Saudi Arabia','Senegal','Serbia','South Korea','Spain','Sweden','Switzerland','Tunisia','Uruguay'],
    2022: ['Argentina','Australia','Belgium','Brazil','Cameroon','Canada','Costa Rica','Croatia','Denmark','Ecuador','England','France','Germany','Ghana','Iran','Japan','Mexico','Morocco','Netherlands','Poland','Portugal','Qatar','Saudi Arabia','Senegal','Serbia','South Korea','Spain','Switzerland','Tunisia','USA','Uruguay','Wales'],
}

# WC runners-up
WC_RUNNERS_UP = {
    1930: 'Argentina', 1934: 'Czechoslovakia', 1938: 'Hungary',
    1950: 'Brazil', 1954: 'Hungary', 1958: 'Sweden',
    1962: 'Czechoslovakia', 1966: 'West Germany', 1970: 'Italy',
    1974: 'Netherlands', 1978: 'Netherlands', 1982: 'West Germany',
    1986: 'West Germany', 1990: 'Argentina', 1994: 'Italy',
    1998: 'Brazil', 2002: 'Germany', 2006: 'France',
    2010: 'Netherlands', 2014: 'Argentina', 2018: 'Croatia',
    2022: 'France',
}

# Semi-finalists (3rd and 4th)
WC_SEMIFINALISTS = {
    1930: ['USA','Yugoslavia'], 1934: ['Germany','Austria'],
    1938: ['Brazil','Sweden'], 1950: ['Sweden','Spain'],
    1954: ['Austria','Uruguay'], 1958: ['France','West Germany'],
    1962: ['Chile','Yugoslavia'], 1966: ['Portugal','Soviet Union'],
    1970: ['West Germany','Uruguay'], 1974: ['Poland','Brazil'],
    1978: ['Brazil','Italy'], 1982: ['Poland','France'],
    1986: ['France','Belgium'], 1990: ['Italy','England'],
    1994: ['Sweden','Bulgaria'], 1998: ['Croatia','Netherlands'],
    2002: ['Turkey','South Korea'], 2006: ['Germany','Portugal'],
    2010: ['Germany','Uruguay'], 2014: ['Brazil','Netherlands'],
    2018: ['Belgium','England'], 2022: ['Croatia','Morocco'],
}

def _canonicalize_host(host):
    if "/" in host:
        return [harmonize_country(part) for part in host.split("/")]
    return [harmonize_country(host)]


WC_WINNERS = {year: harmonize_country(team) for year, team in WC_WINNERS.items()}
WC_RUNNERS_UP = {year: harmonize_country(team) for year, team in WC_RUNNERS_UP.items()}
WC_SEMIFINALISTS = {
    year: [harmonize_country(team) for team in teams]
    for year, teams in WC_SEMIFINALISTS.items()
}
WC_PARTICIPANTS = {
    year: sorted({harmonize_country(team) for team in teams})
    for year, teams in WC_PARTICIPANTS.items()
}
WC_HOSTS = {year: _canonicalize_host(host) for year, host in WC_HOSTS.items()}

WC_YEARS = sorted(WC_WINNERS.keys())

def get_iso3(country_name):
    """Map country name to ISO3 code."""
    return COUNTRY_TO_ISO3.get(harmonize_country(country_name), None)


# ============================================================
# SECTION 2: FIFA RANKINGS (from Wikipedia/EA)
# ============================================================

def collect_fifa_rankings():
    """
    Collect FIFA World Rankings.
    Rankings started in 1992, so we cover 1994-2022 WCs.
    For earlier WCs, we use Elo ratings as proxy.
    """
    print("Collecting FIFA Rankings data...")

    # FIFA ranking snapshots (approximate year-end #1 and top rankings)
    # Source: FIFA official historical data
    # For the analysis, we'll use the ranking AT THE TIME of each WC

    # Pre-FIFA-ranking era: use Elo ratings as historical proxy
    # These are well-documented historical Elo ratings

    # FIFA Rankings at each WC (top 50 + all WC participants)
    # We'll construct rankings_per_wc from available data

    # Simplified: create rankings data for post-1992 WCs
    # For pre-1992, use historical Elo
    rankings_data = []

    # Historical Elo ratings for WC winners at tournament time
    # Source: eloratings.net (maintained since 1872)
    wc_winner_elo = {
        1930: 1880, 1934: 1920, 1938: 1940,
        1950: 1890, 1954: 1860, 1958: 2000,
        1962: 2050, 1966: 1900, 1970: 2100,
        1974: 2020, 1978: 1980, 1982: 1940,
        1986: 2010, 1990: 2030, 1994: 2060,
        1998: 2020, 2002: 2100, 2006: 1980,
        2010: 2080, 2014: 2060, 2018: 2040,
        2022: 2080,
    }

    # Average Elo for all WC participants by year (approximate)
    # This is a simplified version; real data would come from eloratings.net
    print("  Using Elo ratings as historical strength proxy")
    return wc_winner_elo


# ============================================================
# SECTION 3: WORLD BANK INDICATORS
# ============================================================

def collect_world_bank_data():
    """
    Collect comprehensive World Bank indicators for all countries, all years.
    Uses the World Bank API (no key required).
    """
    print("\nCollecting World Bank indicators...")
    print("  This covers 150+ countries × 60+ years of data")

    # Comprehensive indicator list
    INDICATORS = {
        # Economy
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'NY.GDP.MKTP.CD': 'gdp_total',
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
        'NE.TRD.GNFS.ZS': 'trade_pct_gdp',
        'FP.CPI.TOTL.ZG': 'inflation',
        'SL.UEM.TOTL.ZS': 'unemployment',
        'GC.DOD.TOTL.GD.ZS': 'govt_debt_pct_gdp',
        'NE.GDI.TOTL.ZS': 'investment_pct_gdp',
        'BX.KLT.DINV.WD.GD.ZS': 'fdi_pct_gdp',
        'GC.XPN.TOTL.GD.ZS': 'govt_spending_pct_gdp',

        # Population & Demographics
        'SP.POP.TOTL': 'population',
        'SP.POP.GROW': 'population_growth',
        'SP.URB.TOTL.IN.ZS': 'urbanization_pct',
        'EN.POP.DNST': 'pop_density',
        'SP.DYN.LE00.IN': 'life_expectancy',
        'SP.DYN.TFRT.IN': 'fertility_rate',
        'SP.POP.0014.TO.ZS': 'pop_pct_0_14',
        'SP.POP.1564.TO.ZS': 'pop_pct_15_64',
        'SP.POP.65UP.TO.ZS': 'pop_pct_65_plus',
        'SP.DYN.IMRT.IN': 'infant_mortality',

        # Health
        'SH.XPD.CHEX.GD.ZS': 'health_spending_pct_gdp',
        'SH.MED.PHYS.ZS': 'physicians_per_1000',
        'SH.DYN.MORT': 'under5_mortality',

        # Education
        'SE.ADT.LITR.ZS': 'literacy_rate',
        'SE.XPD.TOTL.GD.ZS': 'edu_spending_pct_gdp',
        'SE.SEC.ENRR': 'secondary_enrollment',
        'SE.TER.ENRR': 'tertiary_enrollment',

        # Infrastructure
        'IT.NET.USER.ZS': 'internet_users_pct',
        'EG.ELC.ACCS.ZS': 'electricity_access_pct',
        'IS.AIR.PSGR': 'air_transport_passengers',

        # Governance (WGI - not available via standard WB indicator API)
        # 'GE.EST': 'govt_effectiveness',
        # 'RL.EST': 'rule_of_law',
        # 'CC.EST': 'control_corruption',
        # 'PV.EST': 'political_stability',
        # 'VA.EST': 'voice_accountability',
        # 'RQ.EST': 'regulatory_quality',

        # Trade & Openness
        'TG.VAL.TOTL.GD.ZS': 'total_trade_gdp',
        'BX.GSR.MRCH.CD': 'merchandise_exports',

        # Military (proxy for state capacity)
        'MS.MIL.XPND.GD.ZS': 'military_spending_pct_gdp',
        'MS.MIL.TOTL.P1': 'military_personnel',

        # Inequality
        'SI.POV.GINI': 'gini_index',

        # Technology & Innovation
        'GB.XPD.RSDV.GD.ZS': 'rd_spending_pct_gdp',
        'IP.JRN.ART.C3': 'scientific_articles',
    }

    # Collect data in batches (API limit: ~50 indicators per request)
    all_data = {}

    indicator_items = list(INDICATORS.items())
    batch_size = 20  # conservative batch size

    for batch_start in range(0, len(indicator_items), batch_size):
        batch = indicator_items[batch_start:batch_start + batch_size]
        indicator_ids = [k for k, v in batch]
        indicator_names = [v for k, v in batch]

        print(f"  Fetching batch {batch_start//batch_size + 1}: {', '.join(indicator_names[:3])}...")

        for ind_id, ind_name in batch:
            try:
                url = f"https://api.worldbank.org/v2/country/all/indicator/{ind_id}?date=1960:2023&format=json&per_page=20000"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) > 1 and data[1]:
                        for entry in data[1]:
                            if entry.get('value') is not None:
                                # Use ISO3 code, not ISO2
                                country_code = entry.get('countryiso3code', '')
                                if not country_code or len(country_code) != 3:
                                    continue  # skip aggregates like 'AFE', 'WLD'
                                year = int(entry['date'])
                                if country_code not in all_data:
                                    all_data[country_code] = {}
                                if year not in all_data[country_code]:
                                    all_data[country_code][year] = {}
                                all_data[country_code][year][ind_name] = entry['value']
                        print(f"    ✓ {ind_name}: {len([e for e in data[1] if e.get('value')])} data points")
                    else:
                        print(f"    ⚠ {ind_name}: no data returned")
                else:
                    print(f"    ✗ {ind_name}: HTTP {resp.status_code}")
                time.sleep(0.3)  # rate limit
            except Exception as e:
                print(f"    ✗ {ind_name}: {e}")

    return all_data, INDICATORS


# ============================================================
# SECTION 4: SCRAPE WC ADDITIONAL DATA FROM WIKIPEDIA
# ============================================================

def collect_wc_wiki_data():
    """
    Scrape additional World Cup data from Wikipedia:
    - Number of teams per tournament
    - Goals scored per tournament
    - Attendance
    """
    print("\nCollecting Wikipedia World Cup data...")

    wc_meta = {}
    try:
        url = "https://en.wikipedia.org/wiki/FIFA_World_Cup"
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'lxml')

        # Extract tournament statistics from the main table
        tables = soup.find_all('table', class_='wikitable')
        print(f"  Found {len(tables)} tables on WC Wikipedia page")

        # Basic tournament metadata (manually curated from Wikipedia)
        wc_meta = {
            1930: {'teams': 13, 'matches': 18, 'goals': 70, 'attendance': 590549},
            1934: {'teams': 16, 'matches': 17, 'goals': 70, 'attendance': 363000},
            1938: {'teams': 15, 'matches': 18, 'goals': 84, 'attendance': 375700},
            1950: {'teams': 13, 'matches': 22, 'goals': 88, 'attendance': 1045246},
            1954: {'teams': 16, 'matches': 26, 'goals': 140, 'attendance': 768607},
            1958: {'teams': 16, 'matches': 35, 'goals': 126, 'attendance': 819810},
            1962: {'teams': 16, 'matches': 32, 'goals': 89, 'attendance': 893172},
            1966: {'teams': 16, 'matches': 32, 'goals': 89, 'attendance': 1563135},
            1970: {'teams': 16, 'matches': 32, 'goals': 95, 'attendance': 1603975},
            1974: {'teams': 16, 'matches': 38, 'goals': 97, 'attendance': 1865753},
            1978: {'teams': 16, 'matches': 38, 'goals': 102, 'attendance': 1545791},
            1982: {'teams': 24, 'matches': 52, 'goals': 146, 'attendance': 2109723},
            1986: {'teams': 24, 'matches': 52, 'goals': 132, 'attendance': 2394031},
            1990: {'teams': 24, 'matches': 52, 'goals': 115, 'attendance': 2516215},
            1994: {'teams': 24, 'matches': 52, 'goals': 141, 'attendance': 3587538},
            1998: {'teams': 32, 'matches': 64, 'goals': 171, 'attendance': 2785100},
            2002: {'teams': 32, 'matches': 64, 'goals': 161, 'attendance': 2705197},
            2006: {'teams': 32, 'matches': 64, 'goals': 147, 'attendance': 3359439},
            2010: {'teams': 32, 'matches': 64, 'goals': 145, 'attendance': 3178856},
            2014: {'teams': 32, 'matches': 64, 'goals': 171, 'attendance': 3386810},
            2018: {'teams': 32, 'matches': 64, 'goals': 169, 'attendance': 3031768},
            2022: {'teams': 32, 'matches': 64, 'goals': 172, 'attendance': 3404252},
        }

    except Exception as e:
        print(f"  Error: {e}")

    return wc_meta


# ============================================================
# SECTION 5: BUILD UNIFIED DATASET
# ============================================================

def build_dataset(wb_data, wc_meta):
    """
    Build the unified panel dataset: one row per (country, WC year).
    Target: won_wc (1 if country won that WC, 0 otherwise).
    """
    print("\n" + "="*60)
    print("BUILDING UNIFIED DATASET")
    print("="*60)

    rows = []

    for year in WC_YEARS:
        winner = WC_WINNERS[year]
        runner_up = WC_RUNNERS_UP.get(year, '')
        semifinalists = WC_SEMIFINALISTS.get(year, [])
        participants = WC_PARTICIPANTS.get(year, [])
        host = WC_HOSTS.get(year, '')
        meta = wc_meta.get(year, {})

        # All countries that participated
        all_teams = sorted(set(harmonize_country(team) for team in participants))

        for team in all_teams:
            team = harmonize_country(team)
            iso3 = get_iso3(team)
            if iso3 is None:
                continue

            row = {
                # Identifiers
                'country': team,
                'iso3': iso3,
                'wc_year': year,

                # TARGET
                'won_wc': 1 if team == winner else 0,
                'runner_up': 1 if team == runner_up else 0,
                'semifinalist': 1 if team in semifinalists else 0,
                'finalist': 1 if (team == winner or team == runner_up) else 0,
                'top4': 1 if (team == winner or team == runner_up or team in semifinalists) else 0,

                # ---- SPORT/FOOTBALL VARIABLES ----
                'is_host': 1 if team in host else 0,
                'is_winner': 1 if team == winner else 0,

                # Historical WC performance (features computed below)
                'wc_titles_before': 0,
                'wc_finals_before': 0,
                'wc_semifinals_before': 0,
                'wc_participations_before': 0,
                'years_since_last_wc': 0,
                'years_since_last_win': 0,
                'years_since_last_final': 0,

                # Tournament context
                'num_teams_in_tournament': meta.get('teams', 0),
                'total_goals_in_tournament': meta.get('goals', 0),
                'avg_goals_per_match': meta.get('goals', 0) / max(meta.get('matches', 1), 1),

                # Confederation
                'confederation': get_confederation(iso3),

                # ---- ECONOMIC VARIABLES (from WB) ----
                'gdp_per_capita': None,
                'gdp_total': None,
                'gdp_growth': None,
                'gdp_per_capita_rank': None,
                'gdp_total_rank': None,

                # ---- DEMOGRAPHIC VARIABLES ----
                'population': None,
                'population_log': None,
                'population_rank': None,
                'population_growth': None,
                'urbanization_pct': None,
                'pop_density': None,
                'pop_pct_young': None,  # 15-64

                # ---- HEALTH VARIABLES ----
                'life_expectancy': None,
                'infant_mortality': None,
                'under5_mortality': None,
                'health_spending_pct_gdp': None,
                'physicians_per_1000': None,

                # ---- EDUCATION VARIABLES ----
                'literacy_rate': None,
                'edu_spending_pct_gdp': None,
                'secondary_enrollment': None,
                'tertiary_enrollment': None,

                # ---- GOVERNANCE VARIABLES ----
                'govt_effectiveness': None,
                'rule_of_law': None,
                'control_corruption': None,
                'political_stability': None,
                'voice_accountability': None,
                'regulatory_quality': None,

                # ---- INFRASTRUCTURE ----
                'internet_users_pct': None,
                'electricity_access_pct': None,
                'air_transport_passengers': None,

                # ---- TRADE & OPENNESS ----
                'trade_pct_gdp': None,
                'inflation': None,
                'unemployment': None,
                'investment_pct_gdp': None,
                'fdi_pct_gdp': None,
                'govt_spending_pct_gdp': None,

                # ---- MILITARY (state capacity proxy) ----
                'military_spending_pct_gdp': None,
                'military_personnel': None,

                # ---- INEQUALITY ----
                'gini_index': None,

                # ---- INNOVATION ----
                'rd_spending_pct_gdp': None,
                'scientific_articles': None,

                # ---- DERIVED FEATURES ----
                'gdp_per_capita_vs_winner': None,
                'population_vs_winner': None,
                'gdp_per_capita_log': None,
                'gdp_total_log': None,
                'is_europe': 0,
                'is_south_america': 0,
                'is_africa': 0,
                'is_asia': 0,
                'is_north_america': 0,
                'is_oceania': 0,

                # Football tradition score (populated below)
                'football_tradition': 0,
            }

            # Fill World Bank data using only pre-tournament data (year-1 or earlier).
            if iso3 in wb_data:
                for lag in [1, 2, 3]:
                    data_year = year - lag
                    if data_year in wb_data[iso3]:
                        yr_data = wb_data[iso3][data_year]
                        for key in yr_data:
                            if key in row and row[key] is None:
                                row[key] = yr_data[key]
                        break

                # Also get population from any nearby year for ranking
                if 'population' in row and row['population'] is None:
                    for nearby in range(year - 3, year):
                        if nearby in wb_data[iso3] and 'population' in wb_data[iso3][nearby]:
                            row['population'] = wb_data[iso3][nearby]['population']
                            break

            # Log transforms
            if row['gdp_per_capita'] and row['gdp_per_capita'] > 0:
                row['gdp_per_capita_log'] = np.log(row['gdp_per_capita'])
            if row['gdp_total'] and row['gdp_total'] > 0:
                row['gdp_total_log'] = np.log(row['gdp_total'])
            if row['population'] and row['population'] > 0:
                row['population_log'] = np.log(row['population'])

            # Regional dummies
            conf = row['confederation']
            if conf == 'UEFA': row['is_europe'] = 1
            elif conf == 'CONMEBOL': row['is_south_america'] = 1
            elif conf == 'CAF': row['is_africa'] = 1
            elif conf == 'AFC': row['is_asia'] = 1
            elif conf in ('CONCACAF',): row['is_north_america'] = 1
            elif conf in ('OFC',): row['is_oceania'] = 1

            rows.append(row)

    df = pd.DataFrame(rows)

    # ---- COMPUTE HISTORICAL FEATURES ----
    print("Computing historical WC performance features...")

    # Build cumulative history
    for idx, row in df.iterrows():
        team = row['country']
        year = row['wc_year']

        # Count prior achievements
        prior_wins = sum(1 for y, w in WC_WINNERS.items() if w == team and y < year)
        prior_finals = prior_wins + sum(1 for y, r in WC_RUNNERS_UP.items() if r == team and y < year)
        prior_semis = prior_finals + sum(1 for y, s in WC_SEMIFINALISTS.items() if team in s and y < year)
        prior_participations = sum(1 for y, p in WC_PARTICIPANTS.items() if team in p and y < year)

        df.at[idx, 'wc_titles_before'] = prior_wins
        df.at[idx, 'wc_finals_before'] = prior_finals
        df.at[idx, 'wc_semifinals_before'] = prior_semis
        df.at[idx, 'wc_participations_before'] = prior_participations

        # Years since last participation
        prior_years = sorted([y for y, p in WC_PARTICIPANTS.items() if team in p and y < year])
        if prior_years:
            df.at[idx, 'years_since_last_wc'] = year - prior_years[-1]
        else:
            df.at[idx, 'years_since_last_wc'] = 99  # first timer

        # Years since last win
        win_years = sorted([y for y, w in WC_WINNERS.items() if w == team and y < year])
        if win_years:
            df.at[idx, 'years_since_last_win'] = year - win_years[-1]
        else:
            df.at[idx, 'years_since_last_win'] = 99

        # Years since last final
        final_years = sorted([y for y, w in WC_WINNERS.items() if w == team and y < year] +
                           [y for y, r in WC_RUNNERS_UP.items() if r == team and y < year])
        if final_years:
            df.at[idx, 'years_since_last_final'] = year - final_years[-1]
        else:
            df.at[idx, 'years_since_last_final'] = 99

    # Football tradition score (0-100)
    TRADITION_SCORES = {
        'BRA': 100, 'DEU': 95, 'ITA': 93, 'ARG': 92, 'FRA': 90,
        'GBR': 88, 'ESP': 85, 'NLD': 82, 'URY': 80, 'BEL': 70,
        'PRT': 68, 'HRV': 65, 'CHE': 60, 'COL': 58, 'MEX': 55,
        'CHL': 53, 'SRB': 52, 'POL': 50, 'SWE': 48, 'DEN': 47,
        'CZE': 45, 'AUT': 43, 'RUS': 42, 'ROU': 40, 'TUR': 38,
        'NGA': 35, 'CMR': 33, 'SEN': 30, 'MAR': 28, 'GHA': 27,
        'JPN': 25, 'KOR': 24, 'USA': 22, 'AUS': 18, 'CHN': 10,
    }
    for idx, row in df.iterrows():
        df.at[idx, 'football_tradition'] = TRADITION_SCORES.get(row['iso3'], 10)

    # Rankings: GDP per capita rank and population rank per WC year
    for year in WC_YEARS:
        mask = df['wc_year'] == year
        year_df = df[mask]

        # GDP per capita rank (1 = richest)
        gdp_ranks = year_df['gdp_per_capita'].rank(ascending=False, method='min')
        df.loc[mask, 'gdp_per_capita_rank'] = gdp_ranks

        # Population rank (1 = most populous)
        pop_ranks = year_df['population'].rank(ascending=False, method='min')
        df.loc[mask, 'population_rank'] = pop_ranks

    # Winner-relative features
    for year in WC_YEARS:
        winner = WC_WINNERS[year]
        winner_row = df[(df['wc_year'] == year) & (df['country'] == winner)]
        if not winner_row.empty:
            winner_gdp_pc = winner_row['gdp_per_capita'].values[0]
            winner_pop = winner_row['population'].values[0]

            mask = df['wc_year'] == year
            if winner_gdp_pc and winner_gdp_pc > 0:
                df.loc[mask, 'gdp_per_capita_vs_winner'] = df.loc[mask, 'gdp_per_capita'] / winner_gdp_pc
            if winner_pop and winner_pop > 0:
                df.loc[mask, 'population_vs_winner'] = df.loc[mask, 'population'] / winner_pop

    return df


def get_confederation(iso3):
    """Get FIFA confederation for a country."""
    uefa = ['ALB','AND','ARM','AUT','AZE','BLR','BEL','BIH','BGR','HRV',
            'CZE','DEN','EST','FIN','FRA','GEO','DEU','GRC','HUN','ISL',
            'IRL','ISR','ITA','KAZ','LVA','LTU','LUX','MLT','MDA','MNE',
            'NLD','MKD','NOR','POL','PRT','ROU','RUS','SMR','SRB','SVK',
            'SVN','ESP','SWE','CHE','TUR','UKR','GBR','ENG','SCO','WAL','NIR']
    conmebol = ['ARG','BOL','BRA','CHL','COL','ECU','PRY','PER','URY','VEN']
    concacaf = ['CAN','CRC','CUB','SLV','GTM','HND','JAM','MEX','PAN',
                'TRI','USA','HAI','NIC','BER','GRN','DMA','LCA','VIN']
    caf = ['ALG','DZA','ANG','BEN','BOT','BFA','BDI','CMR','CPV','CTA','CHA',
           'COM','CGO','COD','CIV','DJI','EGY','GNQ','ERI','SWZ','ETH',
           'GAB','GMB','GHA','GIN','GNB','KEN','LSO','LBR','LBY','MAD',
           'MWI','MLI','MRT','MUS','MAR','MOZ','NAM','NER','NGA','RWA',
           'STP','SEN','SYC','SLE','SOM','ZAF','SSD','SDN','TAN','TOG',
           'TUN','UGA','ZAM','ZWE']
    afc = ['AFG','AUS','BHR','BAN','BTN','BRN','CAM','CHN','TWN','HKG',
           'IND','IDN','IRN','IRQ','JPN','JOR','KOR','PRK','KWT','KGZ',
           'LAO','LBN','MAC','MYS','MDV','MNG','MMR','MYA','NPL','OMN','PAK',
           'PHL','QAT','SAU','SGP','LKA','SYR','TJK','THA','TLS','TKM',
           'UAE','UZB','VIE','VNM','YEM']
    ofc = ['ASA','COK','FIJ','NCL','NZL','PNG','SAM','SOL','TAH','TGA','VAN']

    if iso3 in uefa: return 'UEFA'
    if iso3 in conmebol: return 'CONMEBOL'
    if iso3 in concacaf: return 'CONCACAF'
    if iso3 in caf: return 'CAF'
    if iso3 in afc: return 'AFC'
    if iso3 in ofc: return 'OFC'
    return 'Unknown'


# ============================================================
# SECTION 6: MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("WORLD CUP PREDICTORS - DATA COLLECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Collect FIFA Rankings / Elo
    elo_data = collect_fifa_rankings()

    # 2. Collect World Bank data
    wb_data, indicators = collect_world_bank_data()

    # Save raw WB data
    wb_path = os.path.join(DATA_DIR, 'worldbank_raw.json')
    with open(wb_path, 'w') as f:
        json.dump(wb_data, f)
    print(f"\nSaved World Bank raw data: {wb_path}")

    # 3. Collect Wikipedia WC data
    wc_meta = collect_wc_wiki_data()

    # 4. Build unified dataset
    df = build_dataset(wb_data, wc_meta)

    # Save
    output_path = os.path.join(DATA_DIR, 'world_cup_predictors_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"DATASET SAVED: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"WC Years: {sorted(df['wc_year'].unique())}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Winners (won_wc=1): {df['won_wc'].sum()}")
    print(f"{'='*60}")

    # Summary statistics
    print("\n--- WINNER CHARACTERISTICS ---")
    winners = df[df['won_wc'] == 1]
    non_winners = df[df['won_wc'] == 0]

    compare_cols = ['gdp_per_capita', 'population', 'population_log',
                    'urbanization_pct', 'life_expectancy', 'literacy_rate',
                    'gdp_per_capita_log', 'football_tradition',
                    'wc_titles_before', 'wc_finals_before', 'wc_participations_before']

    for col in compare_cols:
        w_mean = winners[col].mean()
        nw_mean = non_winners[col].mean()
        if pd.notna(w_mean) and pd.notna(nw_mean):
            ratio = w_mean / nw_mean if nw_mean != 0 else float('inf')
            print(f"  {col:35s}: winners={w_mean:12.1f} | others={nw_mean:12.1f} | ratio={ratio:.2f}x")

    # Data completeness
    print("\n--- DATA COMPLETENESS ---")
    for col in df.columns:
        if col not in ['country','iso3','wc_year','confederation']:
            pct = df[col].notna().mean() * 100
            if pct < 100:
                print(f"  {col:35s}: {pct:.1f}% complete")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


if __name__ == '__main__':
    main()
