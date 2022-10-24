#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:23:01 2021

@author: l1021338

Class Bend
"""

class Bend:
  """ Store bend parameters associated to a Centerline object
      Params: - bend id
              - index of the upstream inflection point along the centerline
              - index of the downstream inflection point along the centerline
              - age of the bend
              - bend side ('up', 'down', 'unknown')
              - bend is valid
  """

  def __init__(self, bend_id, index_inflex_up, index_inflex_down, age=0,
               side='unknown', isvalid = False):

    self.id = bend_id
    self.bend_evol_id = False
    self.age = age
    self.isvalid = isvalid
    self.side = side
    self.nb_points = index_inflex_down - index_inflex_up + 1

    self.index_inflex_up = index_inflex_up
    self.index_inflex_down = index_inflex_down
    self.index_apex = False
    self.index_max_curv = False

    self.apex_probability = False
    self.apex_probability_smooth = False

    self.pt_middle = False
    self.pt_centroid = False
    self.polygon = False

    self.intersected_section_indexes = False

    # Sinuosity, Length, half-wavelength, Amplitude perpendicular, Amplitude middle
    # individual meander geometry
    self.params = False

    # meander geometry averaged over a given window (computed later)
    self.params_averaged = False

  def __repr__(self):
    return str(self.id)


  # add properties of self and another bend
  # return a new bend with the same id as self
  def __add__(self, bend, nb=1, sinuo_thres=1, apex_proba_ponds=(1.,1.,1.)):
    new_bend = Bend(self.id, self.index_inflex_up, bend.index_inflex_down, self.age,
                    self.side, self.isvalid)
    return new_bend

  def add_intersected_section_index(self, i):
    if not self.intersected_section_indexes:
      self.intersected_section_indexes = []
    self.intersected_section_indexes += [i]