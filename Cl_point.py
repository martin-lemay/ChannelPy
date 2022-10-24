# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:35:32 2019

@author: Martin Lemay

Class Cl_point
"""

import pandas as pd
import numpy as np

class Cl_point:
    """ Centerline point. Store coordinates and all the associated variables
        Params: - channel point id
                - channel point age
                - DataFrame with point coordinates and properties
    """

    def __init__(self, ide, age, dataset):

        self.id = ide
        self.age = age
        self.s = dataset["Curv_abscissa"]
        self.pt = np.array([dataset["Cart_abscissa"], dataset["Cart_ordinate"], dataset["Elevation"]])
        self.data = dataset # curvature, height, velocity, ...

    def __repr__(self):
        return self.pt.__repr__()

    # add the properties of self with those of cl_point
    # return a new Cl_point
    def __add__(self, cl_point):
        array = [self.data[col] + cl_point.data[col] for col in self.data.index]
        data = pd.Series(array, index=self.data.index)
        return Cl_point(self.id, self.age, data)

    # multiply the properties of self by a scalar n
    # return a new Cl_point
    def __mul__(self, n):
        array = [n * self.data[col] for col in self.data.index]
        data = pd.Series(array, index=self.data.index)
        return Cl_point(self.id, self.age, data)
    def __rmul__(self, n):
        return self.__mul__(n)

    def __eq__(self, cl_pt):
        return self.id == cl_pt.id

    def set_curvature(self, curv):
        self.set_property("Curvature", curv)

    def curvature(self):
      return self.get_property("Curvature")

    def velocity(self):
        return self.get_property("Velocity")

    def depth(self):
        return self.get_property("Mean_depth")

    def width(self):
      return self.get_property("Width")

    def velocity_perturbation(self):
      return self.get_property("Vel_perturb")

    def get_property(self, name):
      if name in self.data.index:
        return self.data[name]
      return None

    def set_property(self, name, value):
      self.data[name] = value