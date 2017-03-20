import itertools
import pyLDAvis
import pandas as pd
import re
import simplejson
import seaborn as sns

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda import model, runner
from random import shuffle
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.set_context('talk')

%matplotlib inline


