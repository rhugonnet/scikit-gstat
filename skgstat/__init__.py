from .Variogram import Variogram
from .DirectionalVariogram import DirectionalVariogram
from .SpaceTimeVariogram import SpaceTimeVariogram
from .Kriging import OrdinaryKriging
from .MetricSpace import MetricSpace, MetricSpacePair, ProbabalisticMetricSpace, RasterEquidistantMetricSpace, RasterEquidistantMetricSpace2
from . import interfaces
from . import data
from . import util

# set some stuff
__version__ = '0.6.5'
__author__ = 'Mirko Maelicke <mirko.maelicke@kit.edu>'
__backend__ = 'matplotlib'
