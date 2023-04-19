# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:53:40 2019

@author: Martin Lemay

class CenterlineCollection
"""


import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter

from shapely.geometry import Point, Polygon, LineString
from shapely import affinity, speedups
speedups.enable() # to speed up the geometrical computation with shapely

import dtw

import Centerline
import Bend_evolution
import Isoline
import Section

import centerline_process_function as cpf


class Centerline_collection:
    """ Store all the successive Centerline objects from a single channel-belt
        Params: - file path where to load centerline data
                - spacing between channel points after resampling
                - smoothing distance for Savitsky-Golay filter
                - number of rows to skip in input file
                - start age
                - end age
                - lag between two consecutive inflection point
                - percent of points for smoothing window for apex probability computation
                - sinuosity threshold above which bends are valid
                - width of the channel
                - list of weights for apex probability calculation. Apex probability depends on
                channel point curvature, distance from the middle (amplitude),
                and distance from inflection points
                - boolean to recompute channel point curvature after resampling
                - boolean to interpolate properties after resampling
                - boolean to plot channel point curvature along the centerline
                - boolean to compute the morphometry of the centerline
    """

    def __init__(self, filepath, spacing, smooth_distance,
                 filter_raw = 1, start = -999999, end = 999999,
                 lag=1, nb=1, sinuo_thres=1, width=1,
                 apex_proba_ponds=(1.,1.,1.),
                 compute_curvature=False, interpol_props=True,
                 plot_curvature=False):

        self.centerlines = {} # dictionnary key:Centerline
        self.all_iter = []
        self.bends_evol = [] # list of Bend_evolution objects

        self.data_imported = False
        self.bends_tracking_computed = False
        self.section_lines = False
        self.sections_computed = False
        self.sections = False
        self.real_kinematics_computed = False
        self.apparent_kinematics_computed = False


        # 1. import successive centerlines and create Centerline instances
        self.import_data(filepath, spacing, smooth_distance,
                         filter_raw, start, end, lag, nb, sinuo_thres, width,
                         apex_proba_ponds, compute_curvature, interpol_props,
                         plot_curvature)

        print("Data imported")


    def import_data(self, filepath, spacing, smooth_distance,
                    filter_raw, start, end, lag, nb, sinuo_thres, width,
                     apex_proba_ponds, compute_curvature, interpol_props,
                     plot_curvature, compute_geometry):


        dataset = pd.read_csv(filepath, sep=';')

        for key in dataset["Iteration"].unique().tolist():

            data = dataset[dataset["Iteration"]==key]
            # remove the columns to fit with Centerline input
            # new dataframes to avoid Warning messages
            data1 = data.drop('Iteration', axis=1)
            del(data)
            if "Dist_previous" in data1.columns:
                data = data1.drop('Dist_previous', axis=1)
            else:
                data = data1
            del(data1)
            data.reset_index(drop=True, inplace=True)

            # add width property if not included
            if not "Width" in data.columns:
                data["Width"] = width * np.ones(data.shape[0])

            self.centerlines[key] = Centerline.Centerline(key, data, spacing,
                                                          smooth_distance, lag,
                                                          nb, sinuo_thres,
                                                          apex_proba_ponds,
                                                          compute_curvature,
                                                          interpol_props,
                                                          plot_curvature,
                                                          compute_geometry)

        self.all_iter = np.sort(np.array(list(self.centerlines.keys())))

        self.data_imported = True
        return dataset

    def match_centerlines(self, dmax = np.inf, distance_weight=0.1, vel_perturb_weight=0.4,
                          curvature_weight=0.4,
                          window = 5, pattern="asymmetric"):
      prev_key = self.all_iter[0]
      for k, key in enumerate(self.all_iter):
        if k == 0:
          continue

        lx1 = self.centerlines[key].get_property("Cart_abscissa")
        ly1 = self.centerlines[key].get_property("Cart_ordinate")
        lcurv1 = uniform_filter(self.centerlines[key].get_property("Curvature"), size = window, mode='nearest')
        lvel_perturb1 = uniform_filter(self.centerlines[key].get_property("Vel_perturb"), size = window, mode='nearest')

        lx0 = self.centerlines[prev_key].get_property("Cart_abscissa")
        ly0 = self.centerlines[prev_key].get_property("Cart_ordinate")
        lcurv0 = uniform_filter(self.centerlines[prev_key].get_property("Curvature"), size = window, mode='nearest')
        lvel_perturb0 = uniform_filter(self.centerlines[prev_key].get_property("Vel_perturb"), size = window, mode='nearest')

        if (len(lcurv1)==0) | (len(lcurv0)==0):
          continue

        distance_matrix_vel_pertub = np.zeros((len(lcurv1), len(lcurv0)))
        distance_matrix_dist = np.zeros_like(distance_matrix_vel_pertub)
        distance_matrix_curv = np.zeros_like(distance_matrix_vel_pertub)
        for i, (x1, y1, vel_perturb1, curv1) in enumerate(zip(lx1, ly1, lvel_perturb1, lcurv1)):
          for j, (x0, y0, vel_perturb0, curv0) in enumerate(zip(lx0, ly0, lvel_perturb0, lcurv0)):
            d = np.sqrt((x1-x0)**2+(y1-y0)**2)
            if d > dmax:
              d = 1e9
            distance_matrix_dist[i,j] = d
            distance_matrix_vel_pertub[i,j] = abs(vel_perturb1-vel_perturb0)
            distance_matrix_curv[i,j] = abs(abs(curv1)-abs(curv0))

        if distance_matrix_dist[distance_matrix_dist != 1e9].max() > 0.:
          distance_matrix_dist /= distance_matrix_dist[distance_matrix_dist != 1e9].max()
        if distance_matrix_curv.max() > 0.:
          distance_matrix_curv /= distance_matrix_curv.max()
        if distance_matrix_vel_pertub.max() > 0.:
          distance_matrix_vel_pertub /= distance_matrix_vel_pertub.max()

        distance_matrix = (vel_perturb_weight * distance_matrix_vel_pertub +
                           curvature_weight * distance_matrix_curv +
                           distance_weight * distance_matrix_dist)

        alignment = dtw.dtw(distance_matrix, keep_internals=False, step_pattern=pattern)
        indexes = dtw.warp(alignment, index_reference=True)
        self.set_cl_pts_indexes_in_prev_next_centerlines(key, prev_key, indexes, dmax)
        prev_key = key
      return True

    def set_cl_pts_indexes_in_prev_next_centerlines(self, key, prev_key, indexes, dmax=np.inf):
      self.centerlines[key].index_cl_pts_prev_centerline = np.full(self.centerlines[key].nb_points, np.nan)
      self.centerlines[prev_key].index_cl_pts_next_centerline = [[] for _ in range(self.centerlines[prev_key].nb_points)]
      for index_key, index_prev_key in enumerate(indexes):
        pt1 = self.centerlines[key].cl_points[index_key].pt
        pt0 = self.centerlines[prev_key].cl_points[index_prev_key].pt
        # print(index_prev_key, len(self.centerlines[prev_key].index_cl_pts_next_centerline))
        if cpf.distance(pt1, pt0) < dmax:
          self.centerlines[key].index_cl_pts_prev_centerline[index_key] = index_prev_key
          self.centerlines[prev_key].index_cl_pts_next_centerline[index_prev_key] += [index_key]


    def connect_bends(self, dmax, recompute_apex):
      self.bends_evol = []
      return self.connect_bends_apex(dmax, recompute_apex)
      # return self.connect_bends_centroid(dmax, recompute_apex)


    def connect_bends_apex(self, dmax, bend_evol_validity=5):
      bends_evol = []
      prev_key = 0
      # connect apexes backward through time
      for i, key in enumerate(self.all_iter[::-1]):

          if i == 0:
              bends_evol += [[bend] for bend in self.centerlines[key].bends if bend.isvalid]
              prev_key = key
              continue

          for j, bend in enumerate(self.centerlines[key].bends):

              if not bend.isvalid:
                  continue

              # look for the closest apex
              dist = np.nan * np.zeros(len(bends_evol))
              index = False
              for k, bend_saved in enumerate(bends_evol):

                  # if the last bend_saved was added at the previous key
                  # and is on the same side as bend
                  if (bend_saved[-1].isvalid and
                      bend_saved[-1].age == prev_key and
                      bend_saved[-1].side == bend.side):
                      # compute the distance between apex points
                      dist[k] = cpf.distance(self.centerlines[prev_key].cl_points[bend_saved[-1].index_apex].pt,
                                             self.centerlines[key].cl_points[bend.index_apex].pt)

              # take the index of the minimum distance if this distance is lower than dmax
              dmax2 = dmax
              if np.isfinite(dist).any() and np.nanmin(dist) < dmax2:
                  index = np.nanargmin(dist)

              # a bend is found
              if index:
                  bends_evol[index] += [bend]
              # no bend found
              else:
                  bends_evol += [[bend]]

          prev_key = key

          for bend_evol_id, bends in enumerate(bends_evol):
              bend_indexes = {bend.age:bend.id for bend in bends}
              if len(bend_indexes) > 1:
                  print(bend_indexes)
                  self.centerlines[bend.age].bends[bend.id].bend_evol_id = bend_evol_id
                  self.bends_evol += [Bend_evolution.Bend_evolution(bend_indexes, i, len(bend_indexes)>bend_evol_validity)]

      self.bends_tracking_computed = True
      return True

    def connect_bends_centroid(self, dmax, bend_evol_validity):
        bends_evol = []
        prev_key = 0
        # connect apexes backward through time
        for i, key in enumerate(self.all_iter[::-1]):

            if i == 0:
                for bend in self.centerlines[key].bends:
                    if bend.isvalid:
                        bends_evol += [[bend]]
            else:
                for j, bend in enumerate(self.centerlines[key].bends):

                    if not bend.isvalid:
                        continue

                    # look for the closest apex
                    dist = np.nan * np.zeros(len(bends_evol))
                    index = False
                    for k, bend_saved in enumerate(bends_evol):

                        # if the last bend_saved was added at the previous key
                        # and is on the same side as bend
                        if (bend_saved[-1].isvalid and
                            bend_saved[-1].age == prev_key and
                            bend_saved[-1].side == bend.side):
                            # compute the distance between upstream inflex points (more stable than apex)
                            dist[k] = cpf.distance(bend_saved[-1].pt_centroid, bend.pt_centroid)

                    # take the index of the minimum distance if this distance is lower than dmax
                    if np.isfinite(dist).any() and np.nanmin(dist) < dmax:
                        index = np.nanargmin(dist)

                    # a bend is found
                    if index:
                        bends_evol[index] += [bend]
                    # no bend found, create a new list of bends
                    else:
                        bends_evol += [[bend]]

            prev_key = key

        for bends in bends_evol:
            bend_indexes = {key:bend.id for bend in bends}
            self.bends_evol += [Bend_evolution.Bend_evolution(bend_indexes, i, len(bend_indexes)>bend_evol_validity)]

        self.bends_tracking_computed = True
        return True

    def set_section_lines(self, pts_start, pts_end):
        self.section_lines = []
        for pt_start, pt_end in zip(pts_start, pts_end):
            section_line = LineString((pt_start, pt_end))
            self.section_lines += [section_line]

    def create_section_lines(self, method="from_middle"):
        if method=="from_neighboring_apex":
            self.create_section_lines_from_middle_of_neighboring_apex()
        elif method=="from_middle":
            self.create_section_lines_from_bend("middle")
        elif method=="from_centroid":
            self.create_section_lines_from_bend("centroid")
        else:
            print("Unkown method. Methods are either: \"from_middle\", \"from_centroid\" or \"from_neighboring_apex\"")

    def create_section_lines_from_bend(self, point_name="middle"):
        self.section_lines = []
        for i, bend in enumerate(self.centerlines[self.all_iter[-1]].bends):
            if (not bend.isvalid or (i==0) or (i>len(self.centerlines[self.all_iter[-1]].bends)-2)):
                continue
            key = self.all_iter[-1]
            pt_end = bend.pt_middle
            if (point_name == "centroid"):
                pt_end = bend.pt_centroid
            section_line = LineString((self.centerlines[key].cl_points[bend.index_apex].pt, pt_end))
            self.section_lines += [section_line]
        if (len(self.section_lines) == 0):
            self.section_lines = False

    def create_section_lines_from_neighboring_apex(self):

      self.section_lines = []
      for i, bend in enumerate(self.centerlines[self.all_iter[-1]].bends):
          if (not bend.isvalid or (i==0) or (i>len(self.centerlines[self.all_iter[-1]].bends)-2)):
              continue

          key = self.all_iter[-1]
          prev_bend = self.centerlines[key].bends[i-1]
          next_bend = self.centerlines[key].bends[i+1]

          if (prev_bend.isvalid):
              pt0 = self.centerlines[key].cl_points[prev_bend.index_apex].pt
          else:
              k = prev_bend.index_inflex_up + int((prev_bend.nb_points + 0.5 ) / 2)
              pt0 = self.centerlines[key].cl_points[k].pt

          if (next_bend.isvalid):
              pt1 = self.centerlines[key].cl_points[next_bend.index_apex].pt
          else:
              k = next_bend.index_inflex_up + int((next_bend.nb_points + 0.5 ) / 2)
              pt1 = self.centerlines[key].cl_points[k].pt

          pt_end = (np.array(pt0) + np.array(pt1)) / 2
          section_line = LineString((self.centerlines[key].cl_points[bend.index_apex].pt, pt_end))
          self.section_lines += [section_line]

      if (len(self.section_lines) == 0):
          self.section_lines = False

    # done here because may collect centerline points outside bend_evol
    def find_points_on_sections(self, thres=1, width = 20, depth = 1, flow_dir=np.array([1,0]), cl_collec_id=0):

        if not self.section_lines:
            print("Error: Please first define section lines")
            return False

        self.sections = []
        # for each bend_evol
        for i, section_line in enumerate(self.section_lines):
            # list of isoline instances to store channel locations
            isolines = []
            cl_pt_indexes = []

            # research window area defined by the square whose the section is a diagonal
            line2 = affinity.rotate(section_line, 90) # take the perpendicular
            window = Polygon((np.array(section_line)[0],
                              np.array(line2)[0],
                              np.array(section_line)[1],
                              np.array(line2)[1]))

            # for each centerline
            cl_pts = []
            for key in self.all_iter:
                # for each point of the centerline
                for j, cl_pt in enumerate(self.centerlines[key].cl_points):
                    # if the point is inside the window
                    if window.contains(Point(cl_pt.pt)):

                        if j < len(self.centerlines[key].cl_points)-2:
                            cl_pt2 = self.centerlines[key].cl_points[j+1]
                            cl_line = LineString([cl_pt.pt, cl_pt2.pt])
                            intersect = section_line.intersection(cl_line)
                            # if the intersection exists
                            if not intersect.is_empty:
                                # interpolate channel points properties to the intersection point
                                d = intersect.distance(Point(cl_pt.pt)) / cl_line.length
                                cl_pts += [(cl_pt, cl_pt2)]

                                cl_pt = cl_pt*(1-d) + cl_pt2*d

                                isoline = Isoline.Isoline(key, cl_pt, "Channel")
                                isolines += [isoline]
                                cl_pt_indexes += [j]


            if len(isolines) > thres:
                for k, (isoline, cl_pt_index) in enumerate(zip(isolines, cl_pt_indexes)):
                    isoline.complete_channel_shape(11)

                    # notify bend that is intersected by the section line
                    bend_index = self.get_bend_index_from_cl_point_index(cl_pt_index, isoline.age)
                    self.centerlines[isoline.age].bends[bend_index].add_intersected_section_index(i)

                bend_id = "%s-%s"%(cl_collec_id, bend_index)
                ide = "%s-%s"%(cl_collec_id, i)
                self.sections += [Section.Section(ide, bend_id, section_line.boundary[0].coords[0],
                                  section_line.boundary[1].coords[0], isolines, None, cl_pts[k],
                                  flow_dir)]

        self.sections_computed = True
        return True

    def get_bend_index_from_cl_point_index(self, cl_pt_index, age):
        if (cl_pt_index < self.centerlines[age].bends[0].index_inflex_up):
            return 0
        elif (cl_pt_index > self.centerlines[age].bends[-1].index_inflex_down):
            return len(self.centerlines[age].bends)-1

        for bend_index, bend in enumerate(self.centerlines[age].bends):
            if (cl_pt_index >= bend.index_inflex_up) & (cl_pt_index < bend.index_inflex_down):
                return bend_index
        return bend_index

    def compute_channel_real_kinematics(self, norm_hor=1, norm_vert=1,
                                        write_results=False, filepath=""):
        if self.bends_tracking_computed:
            if write_results:
                fout = open(filepath, "w")
                fout.write("inflex_deltaX;inflex_deltaY;inflex_deltaZ;inflex_deltaMig;")
                fout.write("apex_deltaX;apex_deltaY;apex_deltaZ;apex_deltaMig\n")
                fout.close()

            for bend_evol in self.bends_evol:
                bend_evol.compute_bend_real_kinematics(norm_hor, norm_vert,
                                                       write_results, filepath)
            self.real_kinematics_computed = True
            return True

        print("Error: Please first compute bend tracking")
        return False

    def compute_channel_apparent_kinematics(self, norm_hor=1, norm_vert=1,
                                            write_results=False, filepath=""):
        if self.sections_computed:
            if write_results:
                fout = open(filepath, "w")
                fout.write("Bcb_norm_full;Hcb_norm_full;Bcb_on_Hcb_full;Msb_norm_full;")
                fout.write("Bcb_norm_bend;Hcb_norm_bend;Bcb_on_Hcb_bend;Msb_norm_bend\n")
                fout.close()

            for bend_evol in self.bends_evol:
                bend_evol.compute_bend_apparent_kinematics(norm_hor, norm_vert,
                                                       write_results, filepath)
            self.apparent_kinematics_computed = True
            return True
        print("Error: Please first compute the sections")
        return False


