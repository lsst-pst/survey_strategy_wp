"""
`LynneSim` is a simple tool for estimating LSST survey depth and area trade-offs.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# For fancier plots
try:
    import numpy.ma as ma
    from matplotlib.patches import Ellipse
    import lsst.sims.maf.slicers as slicers
    import lsst.sims.maf.plots as plots
    fancyplot = True
except ImportError:
    fancyplot = False

__all__ = ['LynneSim']


class LynneSim(object):
    """
    Worker class for estimating approximate LSST number of visits, depth etc in a mock proposal-based LSST
    survey campaign.
    """
    def __init__(self, totalNvis=2600000, percentTotal=0.90):
        # Set some constants:
        self.totalNvis = totalNvis  # without snaps (1x30s/visit)
        # Let's say we can play with 90% of these visits:
        self.percentTotal = percentTotal
        self.Nvisits = self.totalNvis * self.percentTotal
        print('Assuming a total number of visits %d and that we will use %f of them:'
              % (self.totalNvis, self.percentTotal))
        print("The number of visits available for use is %d (%.2fM)" % (self.Nvisits, self.Nvisits/1000000))
        self.filters = ['u','g','r','i','z','y']
        # Read in available fields
        self.fields = pd.read_csv('field_list.csv')
        # Prepare to define survey regions
        # Regions defines survey (minisurvey) footprint. Dictionary contains pandas DF defining fields.
        self.regions = {}
        # NvisitsPerField defines expected number of visits per field per region
        self.NvisitsPerField = {}
        # fraction defines expected filter balance per region
        self.fractions = {}
        # filter_visits defines expected number of visits per filter per field per region
        self.filter_visits = {}
        # filter_depths defines expected coadded depth per filter per field per region
        self.filter_depths = {}

    def define_survey_region(self, name, limits=None, NvisitsPerField=None, fractions=None):
        """
        Extract a subset of the fields, within the given spatial limits.
        Note that various regions can overlap!

        Parameters
        ==========
        name: string
            The name of the survey region being specified
        limits: dict
            Limits in ra/dec, galactic b/l and/or ecliptic b/l that define the survey region
        NvisitsPerField: int
            Number of visits per field for this region. If None, this will be calculated later (standard
            behavior). If defined, then this many visits will be reserved per field for this region.
        fractions: dict
            Fraction or number of visits per field per filter (ie. filter balance).
        """
        if limits is None:
            # Use full sky
            limits = {'ra':[0.0,360.0], 'dec':[-90,90],
                      'gl':[0.0,360.0], 'gb':[-90.0,90.0],
                      'el':[0.0,360.0], 'eb':[-90.0,90.0]}

        subset = self.fields
        for coord in limits.keys():
            xmin, xmax = limits[coord][0], limits[coord][1]
            if xmax > xmin:
                plus = 'and'
            else:
                plus = 'or'
            in_this_range = "({x} >= {xmin}) {and_or} ({x} <= {xmax})".format(x=coord, xmin=xmin,
                                                                              and_or=plus, xmax=xmax)
            subset = subset.query(in_this_range)
        self.regions[name] = subset

        if fractions is None:
            # Divide evenly among filters
            onesixth = 1.0/6.0
            self.fractions[name] = {'u':onesixth, 'g':onesixth, 'r':onesixth,
                                    'i':onesixth, 'z':onesixth, 'y':onesixth}
        else:
            fractions_norm = 0
            for f in fractions:
                fractions_norm += fractions[f]
            for f in fractions:
                self.fractions[name] = fractions[f] / fractions_norm
        if NvisitsPerField is not None:
            self.NvisitsPerField[name] = NvisitsPerField

        print('Defined survey region %s' % (name))
        print('  with %d fields in the selected footprint' % (len(self.regions[name])))
        if name in self.NvisitsPerField:
            print('  with %d visits per field' % self.NvisitsPerField[name])
        else:
            print('  (will decide total number of visits per field later)')
        print('  with filter balance: %s' % self.fractions[name])


    def distribute_visits(self):
        """
        Compute Nvis (addition to pandas DF in self.regions[name]) per field, looping over each region.
        """
        # Loop over defined regions, computing number of visits, and depth, in each field.
        # First count the visits which must be assigned to a given region.
        vis_assigned = 0
        assigned_regions = list(self.NvisitsPerField.keys())
        for name in assigned_regions:
            nfields = len(self.regions[name])
            vis_region = self.NvisitsPerField[name] * nfields
            print('Assigned %d visits to %s (%d visits/field * %d fields)' % (vis_region, name, self.NvisitsPerField[name], nfields))
            vis_assigned += vis_region
        print('Assigned %d visits based on NvisitsPerField values to each of %s'
              % (vis_assigned, assigned_regions))
        # Now divide the remaining visits among the other regions evenly.
        vis_remaining = self.Nvisits - vis_assigned
        nfields_remaining = 0
        unassigned_regions = []
        for name in self.regions:
            if name not in assigned_regions:
                nfields_remaining += len(self.regions[name])
                unassigned_regions += [name]
        if len(unassigned_regions) == 0:
            print('There are no fields requiring additional/unassigned number of visits.')
            vis_split = 0
        else:
            vis_per_field = int(np.floor(vis_remaining / nfields_remaining))
            vis_split = vis_per_field * nfields_remaining
            for name in unassigned_regions:
                self.NvisitsPerField[name] = vis_per_field
            print('Assigned %d visits based on even split between %d fields in regions %s (%d per field)'
                  % (vis_split, nfields_remaining, unassigned_regions, vis_per_field))

        vis = vis_split + vis_assigned
        remain = self.Nvisits - vis
        print('This leaves about %d visits out of candidate %d remaining.' % (remain,
                                                                              self.Nvisits))
        print(' (or that these surveys required %.2f of all the original visits.' % (vis / self.totalNvis))

        # Add 'Nvis' column to regions dataframe.
        for name in self.regions:
            self.regions[name] = self.regions[name].assign(Nvis = self.NvisitsPerField[name])
            # Now distribute those visits among the filters, using each region's fractions.
            # Make a visit number dictionary based on the filter fractions:
            filters = list(self.fractions[name].keys())
            values = [int(round(frac*self.NvisitsPerField[name]))
                      for filter, frac in self.fractions[name].items()]
            filter_visits = dict(zip(filters, values))
            # For each filter, add a column:
            for f in filters:
                colname = 'Nvis_' + f
                value = filter_visits[f]
                self.regions[name][colname] = value
            self.filter_visits[name] = filter_visits

        # All regions now have 6 new columns, the number of visits in each filter.
        # Because of the rounding, it's unlikely that Nvis will equal the sum of Nvis_*. Don't worry about it.

    def calculate_metrics(self):
        """
        Compute depth per field, per filter.

        Notes
        =====
        Currently we assume non-overlapping regions...
        """
        # Loop over regions
        for name in self.regions:
            for f in self.fractions[name]:
                nvis = self.regions[name]['Nvis_%s' % f]
                m5 = calculate_depth(nvis, f)
                colname = 'depth_%s' % f
                self.regions[name][colname] = m5

    def plot_sky_map(self, metric=None, clabel=None):
        """
        Plot the desired metric as a sky map.

        Parameters
        ----------
        metric: string
            Name of the metric to be plotted [Nvis, depth]

        Returns
        -------
        plt.figure
        """
        if metric is None:
            pass # Just plot locations, no grayscale for metric.
        else:
            z_min = 100000
            z_max = 0
            for name in self.regions:
                try:
                    z = self.regions[name][metric]
                except KeyError:
                    raise ValueError('Unrecognized metric- must be a column in self.regions.')
                z_min = min(z.min(), z_min)
                z_max = max(z.max(), z_max)

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection="aitoff")
        for i, name in enumerate(self.regions):
            x, y = radec2project(self.regions[name].ra, self.regions[name].dec)
            if metric is None:
                ax.scatter(x, y, alpha=0.7, label=name) # Marker color is assigned automatically
            else:
                z = self.regions[name][metric]
                im = ax.scatter(x, y, c=z, cmap='viridis', vmin=z_min, vmax=z_max)
        plt.grid(True)
        if metric is None:
            plt.legend(loc=(1.0, 0.5))
        else:
            cb = plt.colorbar(im, orientation='horizontal')
            cb.set_label(clabel, fontsize='large')
            #cb.set_clim(z_min, z_max)
        return fig

    def fancy_plot(self):
        """Make a fancier looking sky map of the footprint.
        """
        if not(fancyplot):
            print('Cannot make this fancy plot; MAF plotting utilities unavailable.')
            return None
        # fieldRA / fieldDec are dictionaries - key=prop
        slicer = slicers.OpsimFieldSlicer()
        fignum = None
        colorlist = [[1, 1, 0], [.5, 0, .5], [0, .25, .5], [0, 1, 0],
                      [0, 0, 0], [1, 0, 0], [.5, .5, 1]]
        ci = 0
        colors = {}
        add_planes = True
        for name in self.regions:
            print(name)
            # Modify slicer so we can use it for plotting.
            slicer.slicePoints['ra'] = np.radians(self.regions[name]['ra'])
            slicer.slicePoints['dec'] = np.radians(self.regions[name]['dec'])
            fieldLocs = ma.MaskedArray(data=np.empty(len(self.regions[name]), object),
                                       mask=np.zeros(len(self.regions[name]), bool),
                                       fill_value=-99)
            colors[name] = [colorlist[ci][0], colorlist[ci][1], colorlist[ci][2], 0.4]
            ci += 1
            if ci == len(colorlist):
                ci = 0
            for i in range(len(self.regions[name])):
                fieldLocs.data[i] = colors[name]
            skymap = plots.BaseSkyMap()
            fignum = skymap(fieldLocs, slicer,
                            {'metricIsColor': True, 'bgcolor': 'lightgray', 'raCen': 0, 'figsize': (10, 8),
                             'ecPlane': add_planes, 'mwZone': add_planes},
                            fignum=fignum)
            add_planes = False
        fig = plt.figure(fignum)
        labelcolors = []
        labeltext = []
        for name in self.regions:
            el = Ellipse((0, 0), 0.03, 0.03,
                         fc=(colors[name][0], colors[name][1], colors[name][2]),
                         alpha=colors[name][3])
            labelcolors.append(el)
            labeltext.append(name)
        plt.legend(labelcolors, labeltext, loc=(0.85, 0.9), fontsize='smaller')
        return fig

    def fancy_plot_Nvisits(self, cmap='viridis'):
        """Make a fancier looking sky map of the footprint.
        """
        if not(fancyplot):
            print('Cannot make this fancy plot; MAF plotting utilities unavailable.')
            return None
        # fieldRA / fieldDec are dictionaries - key=prop
        slicer = slicers.OpsimFieldSlicer()
        fignum = None
        # Add a 'survey' that covers the whole sky, and then we put nvisits per survey into it.
        regionlist = list(self.regions.keys())
        self.define_survey_region('_all_nvisits', limits={'dec':[-90, 90]})
        self.regions['_all_nvisits'] = self.regions['_all_nvisits'].assign(Nvis = 0)
        for name in regionlist:
            print(name)
            tmp = self.regions['_all_nvisits']['Nvis'] + self.regions[name]['Nvis']
            tmp = np.where(np.isnan(tmp), 0, tmp)
            self.regions['_all_nvisits']['Nvis'] += tmp
        nvisits = ma.MaskedArray(data=self.regions['_all_nvisits']['Nvis'],
                                 mask=np.zeros(len(self.regions['_all_nvisits']), bool),
                                 fill_value=-99)
        nvisits.mask = np.where(nvisits == 0, True, False)
        # Modify slicer so we can use it for plotting.
        slicer.slicePoints['ra'] = np.radians(self.regions['_all_nvisits']['ra'])
        slicer.slicePoints['dec'] = np.radians(self.regions['_all_nvisits']['dec'])
        skymap = plots.BaseSkyMap()
        fignum = skymap(nvisits, slicer,
                        {'xlabel': 'Nvisits', 'cmap':cmap,
                         'raCen': 0, 'figsize': (10, 8), 'colorMin':0, 'colorMax': 1000,
                         'ecPlane': True, 'mwZone': True},
                        fignum=fignum)
        del self.regions['_all_nvisits']
        fig = plt.figure(fignum)
        return fig

def radec2project(ra, dec):
    x = -1 * np.radians(ra)
    x = np.where(x < -np.pi, x + 2 * np.pi, x)
    y = np.radians(dec)
    return x, y

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
    new_column = pd.DataFrame({column: np.ones(len(df))*value})
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
