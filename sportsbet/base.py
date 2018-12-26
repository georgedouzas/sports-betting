"""
Contains various base classes.
"""

from datetime import datetime as dt


class BaseDataSource:
    """Base class of a data source."""
    
    def download(self):
        """Download the data source."""    
        self.content_ = []
        return self

    def transform(self):
        """Transform the data source."""
    
        return self.content_.copy()

    def download_transform(self):
        """Download and transform the data source."""
        return self.download().transform()


class BaseDataLoader:
    """Base class of a data loader"""

    @property
    def training_data(self):
        """Generate the training data."""
        self.training_time_stamp_ = dt.now().ctime()

    @property
    def fixtures_data(self):
        """Generate the fixtures data."""
        self.fixtures_time_stamp_ = dt.now().ctime()