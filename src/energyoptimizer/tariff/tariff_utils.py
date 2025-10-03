import pandas as pd
import yaml
import os
from typing import Dict, List, Tuple
from datetime import datetime, date
from pandas.tseries.frequencies import to_offset


PANDAS_FREQ_MAP = {'month_last_day_of_month': 'M',
                   'month_first_day_of_month': 'MS',
                   'quarter_last_day_of_quarter': 'Q',
                     'quarter_first_day_of_quarter': 'QS',
                  'week_sunday': 'W-SUN',
                    'week_monday': 'W-MON'}

MIN_DT = pd.Timedelta('1 second')

def load_tariff_yaml(rate_code):
    yaml_path = os.path.join(os.path.dirname(__file__), 'tariffs.yaml')
    with open(yaml_path, 'r') as f:
        tariffs = yaml.safe_load(f)
    return tariffs[rate_code]

def get_index_timedelta_hr(time_index: pd.DatetimeIndex) -> float:
    time_intervals = time_index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600
    return dt

def power_to_energy(power_series: pd.Series) -> pd.Series:
    """
    Convert a power series (kW) to an energy series (kWh).

    Args:
        power_series: Power series with datetime index (positive = import, negative = export)

    Returns:
        Energy series in kWh
    """
    dt_hr = get_index_timedelta_hr(power_series.index)
    energy_series = power_series * dt_hr
    return energy_series

class TariffModel:
    """
    A comprehensive tariff model that handles energy charges, demand charges, and billing cycles.
    
    This class processes tariff information from a YAML file and provides methods to compute
    energy charges, demand charges, and total bills for given power series.
    """
    
    def __init__(self, tariff_file: str, rate_code: str, start_date: date, end_date: date, rate_escalator: float = 0.0, output_freq: str = '15min'):
        """
        Initialize the TariffModel with tariff configuration and date range.
        
        Args:
            tariff_file: Path to the tariff YAML file
            start_date: Start date for the tariff model
            end_date: End date for the tariff model
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.years = range(self.start_date.year, self.end_date.year + 1)
        self.rate_escalator = rate_escalator
        self.output_freq = output_freq
        
        # Load tariff configuration
        if os.path.isabs(tariff_file):
            yaml_path = tariff_file
        else:
            yaml_path = os.path.join(os.path.dirname(__file__), tariff_file)
            
        with open(yaml_path, 'r') as f:
            tariff_config = yaml.safe_load(f)
        
        # Extract rate code (assuming single rate code for now)
        self.rate_code = rate_code
        self.tariff = tariff_config[self.rate_code]
        
        # Get billing data granularity from YAML file
        self.billing_data_granularity = self.tariff.get('billing_data_granularity', '15min')
        
        # Generate time index for the full period using the granularity from YAML
        self.time_index = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=self.output_freq,
            inclusive='both'
        )

        billing_config = self.tariff['billing_cycle']
        billing_freq_code = f"{billing_config['name']}_{billing_config['ending']}"
        self.billing_cycle_freq = PANDAS_FREQ_MAP[billing_freq_code]


        
        # Build tariff components
        self._build_season_period_masks()
        self._build_tariff_timeseries()
        self._build_billing_cycles()
        self._build_demand_charge_categorical_dataframe_and_price_map()

    def _build_rate_series_from_annual_rates(self, mask: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
        full_rate_series = []
        for y in self.years:
            yearly_mask = mask.loc[mask.index.year == y]
            yearly_rates = rates[y]
            yearly_rate_series = yearly_mask.mul(yearly_rates)
            full_rate_series.append(yearly_rate_series)
        full_period_rates = pd.concat(full_rate_series)
        return full_period_rates.sum(axis=1)

    def _build_tariff_timeseries(self):
        """Build the tariff timeseries with energy rates and demand charges."""

        import_tariff = self._build_rate_series_from_annual_rates(self.season_period_masks,
                                                                  self.season_period_import_rates)
        export_tariff = self._build_rate_series_from_annual_rates(self.season_period_masks,
                                                                  self.season_period_export_rates)
        demand_charge = self._build_rate_series_from_annual_rates(self.season_period_masks,
                                                                  self.season_period_demand_rates)
        self.tariff_timeseries = pd.DataFrame({
            'energy_import_rate_kwh': import_tariff,
            'energy_export_rate_kwh': export_tariff,
            'demand_charge_rate_kw': demand_charge
        })

    @staticmethod
    def _series_outer_product(s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
        """Compute the outer product of two series, resulting in a DataFrame."""
        return pd.DataFrame({i: s1 * v for i, v in s2.items()})

    def _build_season_period_masks(self):
        """Build season-period masks that span the entire time period."""
        season_period_masks = {}
        season_period_import_rates = {}
        season_period_export_rates = {}
        season_period_demand_rates = {}
        
        for season in self.tariff['seasons']:
            season_name = season['name']
            months = season['months']
            days = season.get('days', 'all')  # Default to 'all' if not specified
            
            # Build month mask
            month_mask = self.time_index.month.isin(months)
            
            # Build day mask
            if days == 'all':
                day_mask = pd.Series(True, index=self.time_index)
            else:
                # days is a list of day numbers
                day_mask = self.time_index.day.isin(days)
            
            # Combined season mask (months AND days)
            season_mask = month_mask & day_mask
            
            for period in season['periods']:
                period_name = period['name']
                hours = period['hours']
                if hours == 'all':
                    period_mask = pd.Series(True, index=self.time_index)
                else:
                    period_mask = self.time_index.hour.isin(hours)
                
                # Combined season-period mask
                combined_mask = season_mask & period_mask
                season_period_key = f"{season_name}_{period_name}"
                season_period_masks[season_period_key] = combined_mask

                seasonal_rates = season['rates']
                season_period_import_rates[season_period_key] = seasonal_rates['energy_import_rate_kwh'].get(period_name, seasonal_rates['energy_import_rate_kwh'].get('default', 0))
                season_period_export_rates[season_period_key] = seasonal_rates['energy_export_rate_kwh'].get(period_name, seasonal_rates['energy_export_rate_kwh'].get('default', 0))
                season_period_demand_rates[season_period_key] = seasonal_rates['demand_charge_rate_kw'].get(period_name, seasonal_rates['demand_charge_rate_kw'].get('default', 0))

        self.season_period_masks = pd.DataFrame(season_period_masks)

        # Apply rate escalator over the years
        escalators = [(1 + self.rate_escalator) ** i for i in range(len(self.years))]
        escalator_series = pd.Series(escalators, index=self.years)

        import_rates = pd.Series(season_period_import_rates).fillna(0)
        self.season_period_import_rates = self._series_outer_product(import_rates, escalator_series)

        export_rates = pd.Series(season_period_export_rates).fillna(0)
        self.season_period_export_rates = self._series_outer_product(export_rates, escalator_series)

        demand_rates = pd.Series(season_period_demand_rates).fillna(0)
        self.season_period_demand_rates = self._series_outer_product(demand_rates, escalator_series)
            
    
    def _build_demand_charge_categorical_dataframe_and_price_map(self):
        """Build categorical dataframe indicating which periods are subject to demand charges."""
        df = pd.DataFrame(index=self.time_index)
        demand_charge_price_map = {}
        
        # Create billing cycle-specific columns by copying season-period masks
        # for each billing cycle, creating a block diagonal structure
        for cycle_start, cycle_end in self.billing_cycles:
            # Get the month name for this billing cycle
            month_name = cycle_start.strftime('%B').lower()
            year = cycle_start.year
            
            # Find active season-periods in this billing cycle
            billing_cycle_season_periods = self.season_period_masks.loc[cycle_start:cycle_end]
            active_season_periods = billing_cycle_season_periods.columns[billing_cycle_season_periods.sum() > 0]

            # Create columns for each active season-period combination
            for season_period_key in active_season_periods:
                # Create column name for this billing cycle-period combination
                col_name = f"{year}_{month_name}_{season_period_key}"
                
                # Get the mask for this season-period in this billing cycle
                cycle_mask = (self.time_index >= cycle_start) & (self.time_index <= cycle_end)
                billing_cycle_mask = self.season_period_masks[season_period_key] & cycle_mask
                
                df[col_name] = False
                df.loc[billing_cycle_mask, col_name] = True

                rate = self.season_period_demand_rates.loc[season_period_key, year]
                demand_charge_price_map[col_name] = rate
        
        self.demand_charge_categorical_dataframe = df
        self.demand_charge_price_map = pd.Series(demand_charge_price_map).fillna(0)


    def _build_billing_cycles(self):
        """Build billing cycle information based on tariff configuration."""
        """Build billing cycle information based on tariff configuration."""
        billing_freq_offset = to_offset(self.billing_cycle_freq)
        billing_periods = pd.date_range(
            start=self.start_date.round(freq='D') - billing_freq_offset,
            end=self.end_date + billing_freq_offset,
            freq=self.billing_cycle_freq,
            inclusive='both'
        )
        self.billing_cycles = []
        for start, end in zip(billing_periods[:-1], billing_periods[1:]):
            self.billing_cycles.append((start, end - MIN_DT))


    def compute_demand_charge(self, power_series: pd.Series) -> float:
        """
        Compute the demand charge for a given power series.
        
        Args:
            power_series: Power series with datetime index (positive = import, negative = export)
            
        Returns:
            Total demand charge in dollars
        """
        total_demand_charge = 0.0
        aligned_power_series = power_series.resample(self.tariff_timeseries.index.freq, closed='left').max().dropna()


        # For each billing cycle
        for cycle_start, cycle_end in self.billing_cycles:
            if cycle_start > power_series.index[-1] or cycle_end < power_series.index[0]:
                continue
            cycle_power = aligned_power_series.loc[cycle_start:cycle_end]
            if cycle_power.empty:
                continue

            relevant_demand_charge_ts = self.demand_charge_categorical_dataframe.loc[cycle_start:cycle_end, :]
            relevant_demand_charge_ts = relevant_demand_charge_ts.loc[:, (relevant_demand_charge_ts.sum() > 0).values]
            max_power_in_demand_period = relevant_demand_charge_ts.astype(int).mul(cycle_power,axis=0).max()
            period_demand_charges = max_power_in_demand_period * self.demand_charge_price_map

            cycle_demand_charge = period_demand_charges.sum()
            total_demand_charge += cycle_demand_charge
        return total_demand_charge

    def compute_energy_charge(self, power_series: pd.Series) -> float:
        """
        Compute the energy charge for a given power series.
        
        Args:
            power_series: Power series with datetime index (positive = import, negative = export)
            
        Returns:
            Total energy charge in dollars
        """
        # Align power series with tariff timeseries
        energy_series = power_to_energy(power_series)
        aligned_energy_series = energy_series.resample(self.tariff_timeseries.index.freq, closed='left').sum()
        
        # Calculate energy charges
        import_energy = aligned_energy_series.clip(lower=0)  # Only positive (import) power
        export_energy = (-aligned_energy_series).clip(lower=0)  # Only negative (export) power as positive
        
        # Calculate charges
        import_charges = (import_energy * self.tariff_timeseries['energy_import_rate_kwh']).sum()
        export_revenue = (export_energy * self.tariff_timeseries['energy_export_rate_kwh'].fillna(0)).sum()
        
        return import_charges - export_revenue
    
    def compute_total_bill(self, power_series: pd.Series) -> pd.Series:
        """
        Compute the total bill for a given power series.
        
        Args:
            power_series: Power series with datetime index (positive = import, negative = export)
            
        Returns:
            Dictionary with demand_charge, energy_charge, and total_bill
        """
        demand_charge = self.compute_demand_charge(power_series)
        energy_charge = self.compute_energy_charge(power_series)
        total_bill = demand_charge + energy_charge
        
        return pd.Series({
            'demand_charge': demand_charge,
            'energy_charge': energy_charge,
            'total_bill': total_bill
        })
    
    def compute_bill_series(self, power_series: pd.Series) -> pd.DataFrame:
        """
        Compute the bill series for a given power series by billing cycle.
        
        Args:
            power_series: Power series with datetime index (positive = import, negative = export)
            
        Returns:
            DataFrame with columns: demand_charge, energy_charge, total_bill
        """
        bill_series = {}
        
        for cycle_start, cycle_end in self.billing_cycles:
            # Filter power series to this billing cycle
            cycle_power = power_series.loc[cycle_start:cycle_end]
            if cycle_power.empty:
                continue

            bill_series[cycle_start] = self.compute_total_bill(cycle_power)

        return pd.DataFrame.from_dict(bill_series, orient='index')
