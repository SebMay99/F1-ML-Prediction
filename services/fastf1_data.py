from fastf1 import fastf1
import pandas as pd

def collect_race_data(year, round_number):
    """
    Collect all relevant data for a single race
    """
    try:
        # Load race session
        race = fastf1.get_session(year, round_number, 'R')
        race.load()
        
        # Load qualifying session
        try:
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load()
        except:
            quali = None
        
        return race, quali
    
    except Exception as e:
        print(f"  Error loading {year} Round {round_number}: {e}")
        return None, None


def extract_driver_race_data(race, quali, year, round_number):
    """
    Extract per-driver data from a race session
    Returns a DataFrame with one row per driver
    """
    race_results = race.results
    circuit_name = race.event['Location']
    race_name = race.event['EventName']
    race_date = race.event['EventDate']
    
    driver_data = []
    
    for idx, driver in race_results.iterrows():
        driver_dict = {
            # Race identifiers
            'year': year,
            'round': round_number,
            'race_name': race_name,
            'circuit': circuit_name,
            'date': race_date,
            'race_id': f"{year}_{round_number}",
            
            # Driver info
            'driver': driver['Abbreviation'],
            'driver_number': driver['DriverNumber'],
            'team': driver['TeamName'],
            
            # Race results
            'position': driver['Position'] if pd.notna(driver['Position']) else 20,
            'grid_position': driver['GridPosition'] if pd.notna(driver['GridPosition']) else 20,
            'points': driver['Points'] if pd.notna(driver['Points']) else 0,
            'status': driver['Status'],
            
            # Target variables
            'is_winner': 1 if driver['Position'] == 1 else 0,
            'is_podium': 1 if driver['Position'] <= 3 else 0,
            'is_points': 1 if driver['Position'] <= 10 else 0,
            'is_dnf': 1 if 'DNF' in str(driver['Status']) or 'Retired' in str(driver['Status']) else 0,
            
            # Lap data
            'fastest_lap': driver['FastestLap'] if pd.notna(driver['FastestLap']) else None,
        }
        
        # Add qualifying data if available
        if quali is not None:
            try:
                quali_results = quali.results
                driver_quali = quali_results[quali_results['Abbreviation'] == driver['Abbreviation']]
                
                if len(driver_quali) > 0:
                    driver_quali = driver_quali.iloc[0]
                    driver_dict['quali_position'] = driver_quali['Position'] if pd.notna(driver_quali['Position']) else 20
                    driver_dict['q1_time'] = driver_quali['Q1'].total_seconds() if pd.notna(driver_quali['Q1']) else None
                    driver_dict['q2_time'] = driver_quali['Q2'].total_seconds() if pd.notna(driver_quali['Q2']) else None
                    driver_dict['q3_time'] = driver_quali['Q3'].total_seconds() if pd.notna(driver_quali['Q3']) else None
                else:
                    driver_dict['quali_position'] = 20
                    driver_dict['q1_time'] = None
                    driver_dict['q2_time'] = None
                    driver_dict['q3_time'] = None
            except:
                driver_dict['quali_position'] = driver_dict['grid_position']
                driver_dict['q1_time'] = None
                driver_dict['q2_time'] = None
                driver_dict['q3_time'] = None
        else:
            driver_dict['quali_position'] = driver_dict['grid_position']
            driver_dict['q1_time'] = None
            driver_dict['q2_time'] = None
            driver_dict['q3_time'] = None
        
        driver_data.append(driver_dict)
    
    # Find fastest lap driver
    if race_results['FastestLap'].notna().any():
        fastest_lap_time = race_results['FastestLap'].min()
        fastest_driver = race_results[race_results['FastestLap'] == fastest_lap_time]['Abbreviation'].values[0]
        
        for d in driver_data:
            d['is_fastest_lap'] = 1 if d['driver'] == fastest_driver else 0
    else:
        for d in driver_data:
            d['is_fastest_lap'] = 0
    
    return pd.DataFrame(driver_data)


def collect_full_season(year):
    """
    Collect all races from a season
    """
    print(f"\n{'='*70}")
    print(f"Collecting {year} Season")
    print(f"{'='*70}")
    
    schedule = fastf1.get_event_schedule(year)
    all_race_data = []
    
    for idx, event in schedule.iterrows():
        # Skip testing sessions
        if event['EventFormat'] == 'testing':
            continue
        
        round_number = event['RoundNumber']
        print(f"\nRound {round_number}: {event['EventName']}")
        
        race, quali = collect_race_data(year, round_number)
        
        if race is not None:
            driver_data = extract_driver_race_data(race, quali, year, round_number)
            all_race_data.append(driver_data)
            print(f"  ✓ Collected {len(driver_data)} driver records")
        else:
            print(f"  ✗ Failed to collect data")
    
    if all_race_data:
        season_df = pd.concat(all_race_data, ignore_index=True)
        print(f"\n{'='*70}")
        print(f"✓ Season {year} Complete: {len(season_df)} total records")
        print(f"{'='*70}")
        return season_df
    else:
        return pd.DataFrame()


def collect_multiple_seasons(start_year, end_year):
    """
    Collect multiple seasons of data
    """
    all_seasons = []
    
    for year in range(start_year, end_year + 1):
        season_df = collect_full_season(year)
        if len(season_df) > 0:
            all_seasons.append(season_df)
    
    if all_seasons:
        combined_df = pd.concat(all_seasons, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()