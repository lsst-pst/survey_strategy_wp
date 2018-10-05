"""
`LynneSim` is a simple tool for estimating LSST survey depth and area trade-offs.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LynneSim(object):
    """
    Worker class for estimating approximate LSST depth etc in a mock proposal-based LSST survey campaign.
    """
    def __init__(self):
        # Set some constants:
        self.totalNvis = 2600000  # without snaps (1x30s/visit)
        # Let's say we can play with 90% of these visits:
        self.percentTotal = 0.90
        self.Nvisits = self.totalNvis * self.percentTotal
        print("The number of visits available is %d (%.2fM)" % (self.Nvisits, self.Nvisits/1000000))
        self.filters = ['u','g','r','i','z','y']
        # Read in available fields
        self.fields = pd.read_csv('field_list.csv')
        # Prepare to define survey regions
        self.regions = {}
        self.NvisitsPerField = {}
        self.fractions = {}
        self.filter_visits = {}
        self.filter_depths = {}

        return

    def define_survey_region(self, name, limits=None, NvisitsPerField=None, fractions=None):
        """
        Extract a subset of the fields, within the given spatial limits.

        Parameters
        ==========
        name: string
            The name of the survey region being specified
        limits: dict
            Limits in ra/dec, galactic b/l and/or ecliptic b/l that define the survey region
        """
        if limits is None:
            limits = {'ra':[0.0,360.0], 'dec':[-90,90], 'gl':[0.0,360.0], 'gb':[-90.0,90.0], 'el':[0.0,360.0], 'eb':[-90.0,90.0]}
        if fractions is None:
            onesixth = 1.0/6.0
            self.fractions[name] = {'u':onesixth, 'g':onesixth, 'r':onesixth, 'i':onesixth, 'z':onesixth, 'y':onesixth}
        self.NvisitsPerField[name] = NvisitsPerField

        subset = self.fields
        for coord in limits.keys():
            xmin, xmax = limits[coord][0], limits[coord][1]
            if xmax > xmin:
                plus = 'and'
            else:
                plus = 'or'
            in_this_range = "({x} >= {xmin}) {and_or} ({x} <= {xmax})".format(x=coord, xmin=xmin, and_or=plus, xmax=xmax)
            subset = subset.query(in_this_range)
        self.regions[name] = subset

        print("The number of fields in the {0} footprint is {1:d}".format(name, len(subset)))

        return

    def distribute_visits(self):
        """
        Compute Nvis per field in each survey region.
        """
        # Loop over defined regions, computing number of visits, and depth, in each field.
        count = 0
        the_rest = []
        fields_remaining = 0
        for name in self.regions.keys():
            Nfields = len(self.regions[name])
            if self.NvisitsPerField[name] is not None:
                self.regions[name] = add_constant_column(self.regions[name], 'Nvis', self.NvisitsPerField[name])
                this_many = int(round(np.sum(self.regions[name].Nvis)))
                print(this_many," visits allocated to region ", name)
                print(self.NvisitsPerField[name]," visits per ",name," field")
                count += this_many
            else:
                the_rest.append(name)
                fields_remaining += Nfields
        remainder = int(self.Nvisits - count)
        print("Distributing ",remainder," visits among the remaining regions: ",the_rest)

        thisManyVisitsPerField = int(remainder / fields_remaining)
        for name in the_rest:
            Nfields = len(self.regions[name])
            self.NvisitsPerField[name] = thisManyVisitsPerField
            self.regions[name] = add_constant_column(self.regions[name], 'Nvis', self.NvisitsPerField[name])
            this_many = int(round(np.sum(self.regions[name].Nvis)))
            print(this_many," visits allocated to region ", name)
            print(self.NvisitsPerField[name]," visits per ",name," field")

        # All fields in all regions now have an Nvis value - the total number of visits after 10 years.
        # Now distribute those visits among the filters, using each region's fractions.
        for name in self.regions.keys():
            # Make a visit number dictionary based on the filter fractions:
            filters = list(self.fractions[name].keys())
            values = [int(round(frac*self.NvisitsPerField[name])) for filter,frac in self.fractions[name].items()]
            filter_visits = dict(zip(filters, values))
            # For each filter, add a column:
            for filter in filters:
                column = 'Nvis_'+filter
                value = filter_visits[filter]
                self.regions[name] = add_constant_column(self.regions[name], column, value)
            self.filter_visits[name] = filter_visits

        # All regions now have 6 new columns, the number of visits in each filter.
        # Because of the rounding, it's unlikely that Nvis will equal the sum of Nvis_*. Don't worry about it.

        return

    def calculate_metrics(self):
        """
        Compute depth per field, per filter.

        Notes
        =====
        Currently we assume non-overlapping regions...
        """
        # Loop over regions
        for name in self.regions.keys():
            filters = list(self.fractions[name].keys())
            values = [calculate_depth(N,filter) for filter,N in self.filter_visits[name].items()]
            filter_depth = dict(zip(filters, values))
            for filter in filters:
                k = 'Nvis_'+filter
                column = 'depth_'+filter
                value = filter_depth[filter]
                self.regions[name] = add_constant_column(self.regions[name], column, value)


        return

    def plot_sky_map(self, metric=None):
        """
        Plot the desired metric as a sky map.

        Parameters
        ==========
        metric: string
            Name of the metric to be plotted [Nvis, depth]
        """
        if metric is None:
            pass # Just plot locations, no grayscale for metric.

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection="aitoff")
        for name in self.regions.keys():
            x, y = radec2project(self.regions[name].ra, self.regions[name].dec)
            if metric is None:
                ax.scatter(x, y, alpha=0.7, label=name) # Marker color is assigned automatically
            else:
                if metric == 'Nvis':
                    z = self.regions[name].Nvis
                # Need to add Nvis per filter as possible metrics.
                else:
                    try:
                        z = self.regions[name][metric]
                    except:
                        raise ValueError("unrecognized metric {}".format(metric))
                s = ax.scatter(x, y, c=z, cmap='viridis')
                # s.set_clim([0,1000])
        plt.grid(True)
        if metric is None:
            plt.legend()
        else:
            plt.colorbar(s, orientation='horizontal')
        return


def radec2project(ra, dec):
    return (np.radians(ra) - np.pi, np.radians(dec))

def add_constant_column(df, column, value):
    """
    Append a new column of floating point numbers to a dataframe, consisting of the same value, repeated.

    Parameters
    ==========
    df: pandas dataframe
        Table to add a new column to
    column: string
        The name of the new column
    value: float
        The value to populate the column with
    """
    length = len(df)
    new_column = pd.DataFrame({column: np.ones(length)*value})
    df = df.join(new_column)
    return df

def calculate_depth(N, f):
    """
    Given number of visits in a particular filter, compute the 5-sigma limiting magnitude.

    Parameters
    ==========
    N: int
        Number of visits
    f: string
        Name of filter [u,g,r,i,z,y]
    """
    # Current expected performance for single visit 5-sigma depth:
    single_m5 = {'u': 23.98, 'g': 24.91, 'r': 24.42, 'i': 23.97, 'z': 23.38, 'y': 22.47}
    # Scheduler inefficiency varies with band. Calibrated to kraken_2026:
    efficiency_correction = {'u': 0.39, 'g': 0.10, 'r': 0.04, 'i': 0.19, 'z': 0.46, 'y': 0.38}
    return single_m5[f] + 2.5 * np.log10(N) - efficiency_correction[f]
