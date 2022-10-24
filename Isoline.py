# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:58:02 2019

@author: lemaym
"""

import numpy as np
import centerline_process_function as cpf

class Isoline:
    """ List of points of the same age (for instance channel cross-section)
        Params: - age of the points
                - reference Channel_point
                - isoline type (currently only 'Channel')
    """

    def __init__(self, age, cl_pt_ref, isoline_type):

        self.age = age
        self.cl_pt_ref = cl_pt_ref
        self.points = [] # points coordinate according to cl_pt_ref
        self.isoline_type = isoline_type


    def complete_channel_shape(self, nb_pts = 11):

        if self.isoline_type == "Channel":

            # to get an odd number
            if nb_pts % 2 == 0:
                nb_pts += 1
            cl_pt = self.cl_pt_ref

            Xparabol = np.linspace(-1, 1, nb_pts)

            Yparabol = Xparabol*Xparabol * cl_pt.depth()
            Xparabol *= cl_pt.width() / 2.

            self.points = cpf.coords2points(np.array([Xparabol, Yparabol]))

        else:
            print("ERROR: This method can be used only with Channel type isoline")