# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:52:13 2019

@author: Martin Lemay

Class Bend_evolution
"""

import numpy as np

class Bend_evolution:
    """ Store bend indexes in each Centerline that belong to a Bend_evolution
        Params: - indexes of bends in each centerline (dictionnary age:[bend_indexes] in Centerline.bends)
                - id of bend evolution
                - order of bend evolution
                - bend evolution is valid
    """

    def __init__(self, bend_indexes, ide, order, is_valid=False):

      # dictionnary age:[bend_indexes] in Centerline.bends
        self.bend_indexes = bend_indexes
        self.id = ide
        self.order = order
        self.all_iter = np.sort(list(bend_indexes.keys()))
        self.is_valid = is_valid

    def __repr__(self):
       to_return = "last bend id: {} \n".format(self.id)
       to_return += "First iter: {} \n".format(self.all_iter[0])
       to_return += "Last iter: {} \n".format(self.all_iter[-1])
       return to_return

    def is_valid(self, nb):
      return self.all_iter.size >= nb

