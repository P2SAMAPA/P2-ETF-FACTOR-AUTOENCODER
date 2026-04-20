"""
U.S. market calendar utilities.
Provides next trading day based on NYSE calendar.
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

class USMarketCalendar:
    """
    U.S. stock market calendar (NYSE holidays).
    """
    
    def __init__(self):
        self.calendar = USFederalHolidayCalendar()
        self.holidays = self.calendar.holidays(start='2000-01-01', end='2030-12-31')
        self.trading_day = CustomBusinessDay(holidays=self.holidays)
    
    def next_trading_day(self, date=None):
        """Return the next trading day after the given date (default today)."""
        if date is None:
            date = pd.Timestamp.today().normalize()
        else:
            date = pd.Timestamp(date).normalize()
        # Add 1 business day
        return date + self.trading_day
    
    def is_trading_day(self, date=None):
        """Check if the given date is a trading day."""
        if date is None:
            date = pd.Timestamp.today().normalize()
        else:
            date = pd.Timestamp(date).normalize()
        # Check if date is a business day and not a holiday
        return (date.weekday() < 5) and (date not in self.holidays)
