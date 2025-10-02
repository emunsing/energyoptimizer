import numpy as np
import pandas as pd

def ensure_tzinfo(t, tz):
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize(tz)
    return t


def sample_site_data(start_date, end_date, freq, tz='US/Pacific') -> pd.DataFrame:
    """Create sample site data for testing."""
    # Create a simple time series for one month
    np.random.seed(0)

    start_date = ensure_tzinfo(start_date, tz=tz)
    end_date = ensure_tzinfo(end_date, tz=tz)

    index = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Create realistic solar and load profiles
    n_periods = len(index)

    # Solar profile: peak at noon, zero at night, with seasonal variation
    hour_of_day = index.hour + index.minute / 60
    day_of_year = index.dayofyear

    # Seasonal variation: peak in June (~day 172), minimum in December (~day 355)
    # Amplitude varies from ~4 kW in December to ~7 kW in June
    seasonal_amplitude = 5.5 + 1.5 * np.sin((day_of_year - 80) * 2 * np.pi / 365)  # 80 days offset for June peak

    # Daily solar profile with seasonal amplitude
    solar_profile = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * seasonal_amplitude)

    # Load profile: higher during day, lower at night
    load_profile = 3.0 + 2.0 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 0.2, n_periods)
    load_profile = np.maximum(0.5, load_profile)  # Minimum 0.5 kW load

    # Non-circuit load (constant)
    non_circuit_load = np.full(n_periods, 1.0)

    site_data = pd.DataFrame({
        'solar': solar_profile,
        'solar_panel_load': load_profile,
        'main_panel_load': non_circuit_load
    }, index=index)

    return site_data