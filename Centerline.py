# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:34:43 2019

@author: Martin Lemay

Class Centerline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter

from shapely.geometry import Polygon, LineString
from shapely import speedups
speedups.enable() # to speed up the geometrical computation with shapely

import Cl_point
import Bend

import centerline_process_function as cpf


class Centerline:
    """ Store channel centerline as a collection of Cl_point.
        Params: - age of the centerline
                - DataFrame where channel point coordinates and properties come from
                - spacing between channel points after resampling
                - smoothing distance for Savitsky-Golay filter
                - lag between two consecutive inflection point
                - percent of points for smoothing window for apex probability computation
                - sinuosity threshold above which bends are valid
                - list of weights for apex probability calculation. Apex probability depends on
                channel point curvature, distance from the middle (amplitude),
                and distance from inflection points
                - boolean to recompute channel point curvature after resampling
                - boolean to interpolate properties after resampling
                - boolean to plot channel point curvature along the centerline
                - boolean to compute the morphometry of the centerline
    """

    def __init__(self, age, dataset, spacing, smooth_distance, lag=1, nb=1, sinuo_thres=1.05,
                 apex_proba_ponds=(1.,1.,1.),
                 compute_curvature=True, interpol_props=True, plot_curvature=False):

        self.age = age
        self.cl_points = []
        self.nb_points = 0

        self.init_centerline(dataset, age, spacing, smooth_distance, compute_curvature, interpol_props,
                             plot_curvature=plot_curvature)
        self.bends = []
        self.find_bends(lag, nb, sinuo_thres, apex_proba_ponds)
        self.nb_bends = len(self.bends)
        print("Bends found")

        self.index_cl_pts_prev_centerline = False
        self.index_cl_pts_next_centerline = False


    def get_property(self, prop_name):
      try:
        data = []
        for cl_pt in self.cl_points:
          data += [cl_pt.get_property(prop_name)]
        return data
      except Exception as err:
        print("ERROR: %s" %(err))
        return []

    def init_centerline(self, dataset, age, spacing, smooth_distance, compute_curvature=True,
                        interpol_props=False, plot_curvature=False):

        # 1. resample the centerline with a parametric spline function
        nb_pts = int(self.compute_total_curvilinear_length(dataset["Cart_abscissa"], dataset["Cart_ordinate"]) / spacing +1)
        new_points = self.resample_centerline(dataset["Cart_abscissa"], dataset["Cart_ordinate"], nb_pts)
        
        columns = dataset.columns.tolist() + ["Normal_x", "Normal_y"]
        dataset_new = pd.DataFrame(np.zeros((len(new_points[0]), len(columns))),
                                   columns=columns)

        # 2. smooth centerline path
        window = int(float(smooth_distance / spacing)) # number of points

        if window % 2 == 0:
            window += 1 # to be odd
        if window < 5:
            window = 5
        dataset_new["Cart_abscissa"] = savgol_filter(new_points[0], window, polyorder=3, mode='nearest')
        dataset_new["Cart_ordinate"] = savgol_filter(new_points[1], window, polyorder=3, mode='nearest')

        self.compute_curvilinear_abscissa(dataset_new)

        # 2 bis interpolate centerline properties to new points
        # find the 2 closest points in the old centerline, their distances, and interpolate
        if interpol_props:
            self.interpol_properties(dataset_new, dataset)

        # 3. compute and smooth curvatures
        if compute_curvature:
            self.compute_curvature(dataset_new, window)

        self.compute_normal_to_points(dataset_new)

        # Create Centerline object as a collection of cl_Points
        for i, row in dataset_new.iterrows():
            ide = "%s-%s"%(self.age, i)
            self.cl_points += [Cl_point.Cl_point(ide, age, row)]
        self.nb_points = len(self.cl_points)

        if plot_curvature:
            plt.figure()
            plt.plot(dataset_new["Curv_abscissa"], dataset_new["Curvature"], 'k-')
            plt.plot([0, dataset_new["Curv_abscissa"].tolist()[-1]], [0, 0], '--', color='grey')
            plt.show()

        print("Centerline %s initialized"%(self.age))


    def compute_curvilinear_abscissa(self, dataset):

        ls = np.zeros(dataset["Cart_abscissa"].size)
        pt, pt_prev = (0,0), (0,0)
        for i, row in dataset.iterrows():
            pt = (row["Cart_abscissa"], row["Cart_ordinate"])
            if i > 0:
                ls[i] = ls[i-1] + cpf.distance(pt, pt_prev)
            pt_prev = pt

        dataset["Curv_abscissa"] = ls

    def compute_normal_to_points(self, dataset):
        normal = np.array([0.,0.])
        for i, row in dataset.iterrows():
            if i == 0:
                pt_prev = np.array([row["Cart_abscissa"], row["Cart_ordinate"]])
                pt_next = np.array([dataset["Cart_abscissa"][i+1], dataset["Cart_ordinate"][i+1]])
            elif i == dataset.shape[0]-1:
                pt_prev = np.array([dataset["Cart_abscissa"][i-1], dataset["Cart_ordinate"][i-1]])
                pt_next = np.array([row["Cart_abscissa"], row["Cart_ordinate"]])
            else:
                pt_prev = np.array([dataset["Cart_abscissa"][i-1], dataset["Cart_ordinate"][i-1]])
                pt_next = np.array([dataset["Cart_abscissa"][i+1], dataset["Cart_ordinate"][i+1]])

            normal = cpf.perp(pt_next - pt_prev)
            normal /= np.linalg.norm(normal)

            dataset["Normal_x"][i] = normal[0]
            dataset["Normal_y"][i] = normal[1]

    def compute_curvature(self, dataset, smoothing_window):

        for i, row in dataset.iterrows():

            if i > 0 and i < len(dataset["Cart_abscissa"])-1:
                pt1 = (dataset["Cart_abscissa"][i-1], dataset["Cart_ordinate"][i-1])
                pt2 = (row["Cart_abscissa"], row["Cart_ordinate"])
                pt3 = (dataset["Cart_abscissa"][i+1], dataset["Cart_ordinate"][i+1])
                dataset["Curvature"][i] = cpf.compute_curvature(pt1, pt2, pt3)

        # smooth curvature using the Savitzky-Golay filter
        if smoothing_window % 2==0:
            smoothing_window += 1 # to be odd
        if smoothing_window <= 3:
            print("WARNING: curvature smoothing window is 5")
            smoothing_window = 5
        dataset["Curvature"] = savgol_filter(dataset["Curvature"], smoothing_window, polyorder=3)


    def compute_total_curvilinear_length(self, X, Y):
        s = 0
        x_prev, y_prev = 0, 0
        for i, x in enumerate(X):
            y = Y[i]
            if i > 0:
                s += cpf.distance((x, y), (x_prev, y_prev))
            x_prev, y_prev = x, y
        return s


    def resample_centerline(self, x, y, nb_pts=False):
        tck, u = splprep([x, y], s=0)
        if nb_pts:
            u = np.linspace(0., 1., nb_pts)
        return splev(u, tck)

    def get_bend_index_from_cl_pt_index(self, cl_pt_index):
      if cl_pt_index > self.nb_points:
        return np.nan
      for i, bend in enumerate(self.bends):
        if (cl_pt_index >= bend.index_inflex_up) & (cl_pt_index <= bend.index_inflex_down):
          return i
      return np.nan

    def find_closest_point(self, pt, dataset, index=0):
        """ Find the point from the input centerline the closest to the point x,y
            on the new resampled centerline between index and index+1
            Return the index of the closest point in dataset
        """

        d1 = 0
        d2 = 0
        if index < dataset.shape[0]-1:
            d1 = cpf.distance((dataset.loc[index, "Cart_abscissa"],
                               dataset.loc[index, "Cart_ordinate"]),
                               pt)
            d2 = cpf.distance((dataset.loc[index+1, "Cart_abscissa"],
                               dataset.loc[index+1, "Cart_ordinate"]),
                               pt)
        if d1 <= d2:
            return index
        else:
            return index+1


    def interpol_properties(self, dataset_new, dataset):

        j1 = 0 # index of the closest point in dataset
        j2 = 0
        for i, row in dataset_new.iterrows():
            d1 = 1.
            d2 = 0.
            if i == 0 or i == dataset_new.shape[0]-1:
                # copy the properties of the first and last points
                if i == dataset_new.shape[0]-1:
                    j1 = dataset.shape[0]-1

            else:
                # 1. find the closest point in dataset
                pt_new0 = np.array((dataset_new["Cart_abscissa"][i-1], dataset_new["Cart_ordinate"][i-1]))
                pt_new1 = np.array((row["Cart_abscissa"], row["Cart_ordinate"]))
                pt_new2 = np.array((dataset_new["Cart_abscissa"][i+1], dataset_new["Cart_ordinate"][i+1]))

                j1 = self.find_closest_point(pt_new1, dataset, j1)
                pt1 = np.array((dataset["Cart_abscissa"][j1], dataset["Cart_ordinate"][j1]))
                # manage if the closest point is the first or last one
                if j1 == 0:
                    pt0 = np.array((dataset["Cart_abscissa"][j1], dataset["Cart_ordinate"][j1]))
                    pt2 = np.array((dataset["Cart_abscissa"][j1+1], dataset["Cart_ordinate"][j1+1]))
                elif j1 == dataset.shape[0]-1:
                    pt0 = np.array((dataset["Cart_abscissa"][j1-1], dataset["Cart_ordinate"][j1-1]))
                    pt2 = np.array((dataset["Cart_abscissa"][j1], dataset["Cart_ordinate"][j1]))
                else:
                    pt0 = np.array((dataset["Cart_abscissa"][j1-1], dataset["Cart_ordinate"][j1-1]))
                    pt2 = np.array((dataset["Cart_abscissa"][j1+1], dataset["Cart_ordinate"][j1+1]))

                # 2. find the second neighbor point (the one before or after the closest)
                #    by projecting the new point into the former centerline
                pt_proj, j2 = cpf.project_point(pt_new0, pt_new1, pt_new2, pt0, pt1, pt2)

                # 3. interpolate the properties - compute the distances
                d1 = cpf.distance(pt_proj, pt1)
                if j2<0:
                    d2 = cpf.distance(pt_proj, pt0)
                elif j2>0:
                    d2 = cpf.distance(pt_proj, pt2)
                else:
                    d2 = 0

            # 3. interpolate the properties - compute them into the new point
            props = dataset.columns
            denom = d1 + d2
            if denom == 0:
                d1 = 0.5
                d2 = 0.5
                denom = 1.
            
            for prop in props:
                if prop in ("Curv_abscissa", "Cart_abscissa", "Cart_ordinate"):
                    continue

                if (j1+j2 < dataset[prop].size):
                    dataset_new.loc[i, prop] = (d1 * dataset[prop][j1] +
                                                d2 * dataset[prop][j1+j2]) / denom
                else:
                    dataset_new.loc[i, prop] = dataset[prop][j1]
                    
                if (self.age == 1000) & (prop == "Elevation"):
                  print(dataset[prop][j1])
                  print(dataset[prop][j1+j2])
                  print(dataset_new[prop][i])

        return True


    def find_inflexion_points(self, lag=1, window = 5):
        """ Find all inflection points from the object Centerline
            Return the list of inflection point indexes
        """
        # apply average filter of the curvature to smooth local variations
        curvature = np.zeros(len(self.cl_points))
        for i, cl_pt in enumerate(self.cl_points):
            curvature[i] = cl_pt.curvature()
        curvature = uniform_filter(curvature, size = window, mode='nearest')

        inflex_pts = [0] # add the first point of the centerline
        prev_curv = curvature[0]
        for i, curv in enumerate(curvature):
            if (i > 0 and curv * prev_curv < 0):

                if (i > inflex_pts[-1]+lag):
                    inflex_pts += [i]
                else:
                    if len(inflex_pts) > 1:
                        inflex_pts.remove(inflex_pts[-1])

            prev_curv = curv

        # add the last point of the centerline
        if len(self.cl_points)-1 > inflex_pts[-1]+lag:
            inflex_pts += [len(self.cl_points)-1]
        else:
            inflex_pts[-1] = len(self.cl_points)-1
        return inflex_pts

    def find_bends(self, lag=1, nb=1, sinuo_thres=1.001, apex_proba_ponds=(1.,1.,1.)):

        inflex_pts_index = self.find_inflexion_points(lag)
        prev_inflex_index = inflex_pts_index[0]
        for i, inflex_index in enumerate(inflex_pts_index):
            if i==0:
              continue
            bend_index = i-1
            bend = Bend.Bend(bend_index-1, prev_inflex_index, inflex_index, self.age)
            self.bends += [bend]
            self.get_bend_side(bend_index)
            self.check_if_bend_is_valid(bend_index, sinuo_thres)
            self.find_bend_apex(bend_index, nb, apex_proba_ponds)
            prev_inflex_index = inflex_index
        return True

    # work in progress
    def gather_consecutive_invalid_bends(self, sinuo_thres=1.05):
        new_bends = []
        for i, bend in enumerate(self.bends):

            bend.id = len(new_bends) # update bend id
            new_bends += [bend]
            if self.check_if_bend_is_valid(i, sinuo_thres):
                continue

            while i+1 < len(self.bends) and not self.bends[i+1].isvalid:
                new_bends[-1] = new_bends[-1] + self.bends[i+1]
                i += 1
        self.bends_filtered = new_bends
        print("bends filtered")

    # work in progress
    def filter_bends(self, nb=1, sinuo_thres=1):

        k = 0
        for i, bend in enumerate(self.bends):

            # if k>0, bend already gathered with bend i-1
            if k > 0:
                k -= 1
                continue

            # if the bend i is valid it is saved
            if bend.isvalid:
                self.bends_filtered += [bend]
            else:
                # look for the next valid bend
                k = 1
                while i+k < len(self.bends) and not self.bends[i+k].isvalid:
                    k += 1

                if i == 0:
                    self.bends_filtered += [self.bends[i]]
                    for j in range(1, k+1):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]
                else:
                    self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i]
                    # if the last bend is not valid, add to it to the previous one and continue the loop
                    if i+k == len(self.bends):
                        continue

                # if k is even, or the last bends are not valid
                # add all bends (until the next valid one included) to the last valid bend
                if (k%2!=0) or (i+k == len(self.bends)-1 and self.bends[i+k].isvalid):
                    for j in range(1,k+1):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]

                # if k is odd, means that the 2 consecutive valid bends are not by the same side
                elif k%2==0:
                    # get the middle
                    cl_pt_apex0 = self.cl_points[self.bends_filtered[-1].index_apex]
                    cl_pt_apex1 = self.cl_points[self.bends[i+k].index_apex]
                    best_s = (cl_pt_apex0.s + cl_pt_apex1.s)/2
                    # get the inflexion point the closest from the middle

                    ls_inflex = np.array([best_s - self.cl_points[self.bends[i+j].index_inflex_up].s for j in range(k)])
                    n = ls_inflex.argmin()

                    # gather the last valid bend until those until the middle
                    for j in range(1,n):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]
                    # gather the bend in the middle the next ones until the next valid one included
                    self.bends_filtered += [self.bends[i+n]]
                    for j in range(n+1,k+1):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]

        return True

    def compute_bend_apex_probability(self, bend_index, curvature_weight = 1.,
                                      amplitude_weight = 1., length_weight = 1.):
        bend = self.bends[bend_index]
        # renormalization of weights
        tot = curvature_weight + amplitude_weight + length_weight
        if (tot != 1.):
            curvature_weight /= tot
            amplitude_weight /= tot
            length_weight /= tot

        bend.apex_probability = np.full((len(self.cl_points[bend.index_inflex_up:bend.index_inflex_down+1])), np.nan)
        curv_max = 0.
        max_ampl = 0.
        for i, cl_pt in enumerate(self.cl_points[bend.index_inflex_up:bend.index_inflex_down+1]):
            if (abs(cl_pt.curvature()) > curv_max):
                curv_max = abs(cl_pt.curvature())
                bend.index_max_curv = i
            ampl = self.compute_bend_amplitude(bend_index, cl_pt.pt, "middle")

            if ampl > max_ampl:
                max_ampl = ampl

        for i, cl_pt in enumerate(self.cl_points[bend.index_inflex_up:bend.index_inflex_down+1]):
            p = curvature_weight * (abs(cl_pt.curvature()) / curv_max) # curvature
            p += amplitude_weight * self.compute_bend_amplitude(bend_index, cl_pt.pt, "middle") / max_ampl # amplitude
            d = i / ((bend.index_inflex_down-bend.index_inflex_up+1) / 2.) # length from inflection points
            if (d > 1.):
                d = 2.-d
            p += length_weight * d
            bend.apex_probability[i] = p

    def check_if_bend_is_valid(self, bend_index, sinuo_thres = 1.05):
        bend = self.bends[bend_index]
        cl_pt_inflex_up = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down = self.cl_points[bend.index_inflex_down]
        lentgh = abs(cl_pt_inflex_down.s - cl_pt_inflex_up.s)
        d_inflex = cpf.distance(cl_pt_inflex_up.pt, cl_pt_inflex_down.pt)
        sinuo = 1.
        if d_inflex > 0.:
            sinuo = lentgh / d_inflex

        if sinuo < sinuo_thres:
            bend.isvalid = False
        else:
          bend.isvalid = True
        return bend.isvalid

    def get_bend_side(self, bend_index):
        bend = self.bends[bend_index]
        curv = 0
        for cl_pt in self.cl_points[bend.index_inflex_up:bend.index_inflex_down+1]:
            curv += cl_pt.curvature()
        if curv > 0:
            bend.side = "up"
        else:
            bend.side = "down"
        return bend.side

    # apex from apex probability
    def find_bend_apex(self, bend_index, nb_frac=10, apex_proba_ponds=(1.,1.,1.)):
        bend = self.bends[bend_index]
        self.compute_bend_apex_probability(bend_index, apex_proba_ponds[0], apex_proba_ponds[1],
                                           apex_proba_ponds[2])
        nb_frac /= 100
        nb = int(np.round(nb_frac * bend.apex_probability.size))
        if nb == 0:
            nb = 1

        bend.apex_probability_smooth = uniform_filter(bend.apex_probability, size = nb, mode='nearest')

        # get the index of max probability
        bend.index_apex = bend.index_inflex_up + np.nanargmax(bend.apex_probability_smooth)
        return True

    def compute_bend_amplitude(self, bend_index, pt, kind = 'middle'):
        bend = self.bends[bend_index]
        cl_pt_inflex_up = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down = self.cl_points[bend.index_inflex_down]

        if kind == 'perpendicular':
            k = (((cl_pt_inflex_down.pt[0] - cl_pt_inflex_up.pt[0]) *
                  (pt[0] -cl_pt_inflex_up.pt[0]) +
                  (cl_pt_inflex_down.pt[1] - cl_pt_inflex_up.pt[1]) *
                  (pt[1] - cl_pt_inflex_up.pt[1])))
            den = ((cl_pt_inflex_down.pt[0] - cl_pt_inflex_up.pt[0])**2 +
                  (cl_pt_inflex_down.pt[1] - cl_pt_inflex_up.pt[1])**2)

            if den != 0:
                k /= den
            else:
                k = 0 # the 2 points are superposed
        else:
            k = 0.5
        pt_proj = cpf.compute_colinear(cl_pt_inflex_up.pt, cl_pt_inflex_down.pt, k)
        amplitude = cpf.distance(pt_proj, pt)
        return round(amplitude, 4)

    def compute_bend_polygon(self, bend_index, compute_centroid=False):
        bend = self.bends[bend_index]
        pts = [cl_pt.pt for cl_pt in self.cl_points[bend.index_inflex_up:bend.index_inflex_down+1]]
        if len(pts)>2:
            bend.polygon = Polygon(pts)
            if compute_centroid:
                bend.pt_centroid = np.array(bend.polygon.centroid)
        elif len(pts) == 2:
            bend.polygon = LineString(pts)
            if compute_centroid:
                bend.pt_centroid = np.array(bend.polygon.centroid)
        else:
            bend.isvalid = False

    def compute_geometry_leopold(self):

        prev_bend = False
        bend = False
        for i, next_bend in enumerate(self.bends):
            if i > 1:
                pt_apex_prev = self.cl_points[prev_bend.index_apex]
                pt_apex = self.cl_points[bend.index_apex]
                pt_apex_next = self.cl_points[next_bend.index_apex]
                bend.params["Wavelength_Leopold"] = cpf.distance(pt_apex.pt, pt_apex_next.pt)
                pt_proj = cpf.project_perpendicularly(pt_apex.pt, pt_apex_prev.pt, pt_apex_next.pt)
                bend.params["Amplitude_Leopold"] = cpf.distance(pt_apex.pt, pt_proj)

            prev_bend = bend
            bend = next_bend


    def morphometry(self, window_size, leopold=True):

        props = ("Sinuosity", "Length", "Half_Wavelength",
                 "Amplitude_perp", "Amplitude_middle",
                 "Amplitude_Leopold", "Wavelength_Leopold")
        mean_values = pd.Series(np.zeros(len(props)), index=props)
        nb_bends = 0

        # compute individual bend geometry
        for i, bend in enumerate(self.bends):
            self.bend_morphometry(i)
        
        # compute individual bend gemetry according to leopold methology 
        if leopold:
            self.compute_geometry_leopold()
        
        # compute average bend geometry over window and mean values over the centerline
        for i, bend in enumerate(self.bends):
            bend.params_averaged = pd.Series(np.zeros(len(props)), index=props)

            cl_ptmin = self.cl_points[bend.index_inflex_up]
            smin = cl_ptmin.s - window_size
            smax = cl_ptmin.s + window_size
            jmin = i
            while (cl_ptmin.s > smin and jmin>0):
                jmin -= 1
                cl_ptmin = self.cl_points[self.bends[jmin].index_inflex_up]

            jmax = i
            cl_ptmax = self.cl_points[bend.index_inflex_down]
            while (cl_ptmax.s < smax and jmax < len(self.bends)-1):
                jmax += 1
                cl_ptmax = self.cl_points[self.bends[jmax].index_inflex_down]
            
            bend.params_averaged["Length"] = abs(cl_ptmax.s-cl_ptmin.s)
            d = cpf.distance(cl_ptmin.pt, cl_ptmax.pt)
            if d > 0:
                bend.params_averaged["Sinuosity"] = bend.params_averaged["Length"] / d

            nb = 0
            for j in np.arange(jmin, jmax+1):
                nb += 1
                for prop in props:
                    if j == jmin:
                        bend.params_averaged[prop] = 0
                    if np.isfinite(self.bends[j].params[prop]):
                        bend.params_averaged[prop] += self.bends[j].params[prop]

            for prop in props:
                bend.params_averaged[prop] /= nb

                if np.isfinite(bend.params[prop]):
                    mean_values[prop] += bend.params[prop]
                    nb_bends += 1

        mean_values /= nb_bends
        mean_values["Sinuosity"] = abs(self.cl_points[-1].s-self.cl_points[0].s) / cpf.distance(self.cl_points[0].pt, self.cl_points[-1].pt)

        return mean_values

    def bend_morphometry(self, bend_index):
        bend = self.bends[bend_index]
        cl_pt_inflex_up = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down = self.cl_points[bend.index_inflex_down]

        # Sinuosity, Length, half-wavelength, Amplitude perpendicular, Amplitude middle
        bend.params = pd.Series(np.nan*np.zeros(7), index=("Sinuosity", "Length", "Half_Wavelength",
                                                           "Amplitude_perp", "Amplitude_middle",
                                                           "Amplitude_Leopold", "Wavelength_Leopold"))

        # Sinuosity, Length, half-wavelength, Amplitude perpendicular, Amplitude middle
        bend.params_averaged = pd.DataFrame(np.nan*np.zeros((2,7)),
                                            columns=bend.params.index)


        bend.params["Length"] = abs(cl_pt_inflex_down.s - cl_pt_inflex_up.s)

        d_inflex = cpf.distance(cl_pt_inflex_up.pt, cl_pt_inflex_down.pt)
        bend.params["Half_Wavelength"] = d_inflex
        if d_inflex > 0:
            bend.params["Sinuosity"] = bend.params["Length"] / d_inflex
        else:
            bend.params["Sinuosity"] = 1

        # compute the amplitudes
        if (bend.index_apex):
            cl_pt_apex = self.cl_points[bend.index_apex]
            bend.params["Amplitude_perp"] = self.compute_bend_amplitude(bend_index, cl_pt_apex.pt, kind="perpendicular")
            bend.params["Amplitude_middle"] = self.compute_bend_amplitude(bend_index, cl_pt_apex.pt, kind="middle")

    def save_morphometry_results(self, workdir, delimiter=';'):

        props = ["Sinuosity_W1",          "Sinuosity_W2",
                 "Wavelength_Leopold_W1", "Wavelength_Leopold_W2",
                 "Amplitude_Leopold_W1",  "Amplitude_Leopold_W2",
                 "Half_Wavelength_W1",    "Half_Wavelength_W2",
                 "Amplitude_middle_W1",   "Amplitude_middle_W2"]

        data = pd.DataFrame(np.nan*np.zeros((len(self.bends), 11)),
                            columns = ["Bend_ID"]+props)

        for i, bend in enumerate(self.bends):
            data["Bend_ID"][i] = bend.id
            for prop in props:
                data[prop][i] = bend.params_averaged[prop[:-3]]

        data.to_csv(workdir+"morphometry.csv", sep=delimiter, index=False, float_format='%.2f', mode='a')

