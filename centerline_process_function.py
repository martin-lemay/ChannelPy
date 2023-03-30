# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:40:31 2019

@author: Martin Lemay
"""


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from scipy import interpolate
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter



def import_data(filepath, filter_raw = 1, start = -999999, end = 999999):

    dataset = {}
    fin = open(filepath, 'r')
    header = fin.readline().split(';')

    for l in fin:
        line = l.split(';')

        if line[0] not in dataset.keys():
            dataset[line[0]] = [[] for i in range(len(line)-1)]

        for i in range(len(line)-1):
            if float(line[filter_raw+1]) >= start and float(line[filter_raw+1]) <= end:
                if line[1+i].endswith('\n') or line[1+i].endswith(' '):
                    value = float(line[i+1][:-1])
                else:
                    value = float(line[i+1])
                dataset[line[0]][i] += [value]

    fin.close()
    return dataset, header

# create the dataset as Flumy csv file
def create_dataset_from_xy(X, Y):
    data = np.zeros((X.size, 5))
    s = 0
    x_prev, y_prev = 0, 0
    for i, x in enumerate(X):
        y = Y[i]
        data[i, 1] = x
        data[i, 2] = y
        if i == 0:
            data[i, 0] = 0 # curvilinear abscissa
        else:
            # curvilinear abscissa
            s += distance((x, y), (x_prev, y_prev))
            data[i, 0] = s

            # curvature commputation
            if i == 0 or i == X.size-1:
                data[i, 4] = 0 # curvature
            else:
                i_min = max(i-1, 0)
                i_max = min(i+1, X.size-1)
                pt1 = (X[i_min], Y[i_min])
                pt3 = (X[i_max], Y[i_max])

                pt2 = (x, y)
                data[i, 4] = compute_curvature(pt1, pt2, pt3)
        x_prev, y_prev = x, y

    dataset = pd.DataFrame(data, columns=("Curv_abscissa", "Cart_abscissa", "Cart_ordinate",
                                                  "Elevation", "Curvature"))
    return dataset

def filter_dataset(filepath, keys, raw=0, start=-999999, end=999999):

    fin = open(filepath, 'r')
    fout = open(filepath[:-4] + "_filtered.csv", 'w')
    head = True
    for line in fin:
        if head:
            fout.write(line)
            head = False
            continue
        val = line.split(';')

        if keys and val[0] in keys:
            if float(val[raw+1]) >= start and float(val[raw+1]) <= end:
                fout.write(line)

    fin.close()
    fout.close()

    return True

def points2coords(pts):
    coords = np.zeros((len(pts[0]), len(pts)))
    for i, pt in enumerate(pts):
        for j in range(len(pt)):
            coords[j, i] = pt[j]
    return coords

def clpoints2coords(cl_pts):
    coords = np.zeros((len(cl_pts[0].pt), len(cl_pts)))
    for i, cl_pt in enumerate(cl_pts):
        for j in range(len(cl_pt)):
            coords[j, i] = cl_pt.pt[j]
    return coords

def coords2points(coords):
    pts = []
    if coords.shape[0] == 2 or coords.shape[0] == 3:
        dim = coords.shape[0]
        if dim == 2:
            for x,y in zip(coords[0], coords[1]):
                pts += [np.array([x, y])]
        else:
            for x,y,z in zip(coords[0], coords[1], coords[2]):
                pts += [np.array([x, y, z])]

    elif coords.shape[1] == 2 or coords.shape[1] == 3:
        dim = coords.shape[1]
        for coord in coords:
            if dim == 2:
                pts += [coord]
            else:
                pts += [coord]
    else:
        print("Error: bad coordinates format")
    return pts


def compute_colinear(pt1, pt2, k):
    x = pt1[0] + k * (pt2[0] - pt1[0])
    y = pt1[1] + k * (pt2[1] - pt1[1])
    return (x, y)


def distance(pt1, pt2):
    if (type(pt1) == list) | (type(pt1) == tuple):
        pt1 = np.array(pt1)
    if (type(pt2) == list) | (type(pt2) == tuple):
        pt2 = np.array(pt2)

    while pt1.size != pt2.size:
        if pt1.size > pt2.size:
            pt2 = np.append(pt2, 0.)
        elif pt2.size > pt1.size:
            pt1 = np.append(pt1, 0.)

    d = np.linalg.norm(pt2 - pt1)
    return round(d, 4)

def perp(vec) :
    vec_new = np.empty_like(vec)
    vec_new[0] = -vec[1]
    vec_new[1] = vec[0]
    return vec_new

def seg_intersect(pt11,pt12, pt21,pt22) :
    da = pt12-pt11
    db = pt22-pt21
    dp = pt11-pt21
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    if denom.astype(float) != 0:
        return (num / denom.astype(float))*db + pt21
    else:
        return np.zeros(2, dtype=bool)

def project_point(pt_new0, pt_new1, pt_new2, pt0, pt1, pt2):

    # vector along which to project pt_new1
    pt_new12 = (pt_new2 - pt_new0)
    pt_new12 = pt_new1 + perp(pt_new12)

    # projection onto the segment pt0, pt1
    pt_proj0 = seg_intersect(pt_new1,pt_new12, pt0,pt1)
    # projection onto the segment pt2, pt1
    pt_proj2 = seg_intersect(pt_new1,pt_new12, pt2,pt1)

    # keep the closer pojected point when they exist
    if pt_proj0.any() and pt_proj2.any():
        d = distance(pt_new1, pt_proj0) - distance(pt_new1, pt_proj2)
        if d < 0:
            j2 = -1
            pt_proj = pt_proj0
        else:
            j2 = 1
            pt_proj = pt_proj2
    elif pt_proj0.any():
        j2 = -1
        pt_proj = pt_proj0
    elif pt_proj2.any():
        j2 = 1
        pt_proj = pt_proj2
    else:
        pt_proj = pt1
        j2 = 0
        print("WARNING: Error when projecting the point to the former centerline")
    return pt_proj, j2


def smooth_trajec(l_pt, input_ages, output_ages, window=2, resample_curve=False):

  l_pt_interp = l_pt
  dim = 2
  y_interp = []
  for i in range(dim):
      y = []
      for pt in l_pt:
        y += [pt[i]]

      y_interp += [savgol_filter(y, 9, window)]

      if resample_curve:
          tck = interpolate.splrep(input_ages, y_interp[i], s=0)
          y_interp[i] = interpolate.splev(output_ages, tck)

  l_pt_interp = coords2points(np.array(y_interp))
  return l_pt_interp

def get_MP(dir_trans = np.array((1., 0.)), ref = np.array((1., 0.))):
  dir_trans /= np.linalg.norm(dir_trans)
  ref /= np.linalg.norm(ref)
  if (np.dot(dir_trans, ref) < 0.):
    dir_trans *= -1.

  cos = np.dot(dir_trans, ref)
  teta = np.arccos(cos)
  det = np.linalg.det((dir_trans, ref))
  if det < 0:
    teta = np.pi-teta
  sin = np.sin(teta)

  MP = np.array([[cos, sin],
                 [-sin, cos]])
  return MP

def compute_point_displacements(l_pt, dir_trans = np.array((1., 0.)), ref = np.array((1., 0.))):
  # compute change-of-basis matrix
  MP = get_MP(dir_trans, ref)

  # compute displacement
  local_disp = np.nan*np.zeros((len(l_pt)-1, 4)) # dX, dY, dZ, dMig
  whole_disp = np.nan*np.zeros(4) # deltaX, deltaY, deltaZ, deltaMig

  pt1 = l_pt[0]
  for i, pt2 in enumerate(l_pt):
      if i > 0:
          disp = pt2 - pt1
          disp2 = np.dot(MP, disp)

          local_disp[i-1, 0] = disp2[0]
          local_disp[i-1, 1] = disp2[1]
          if len(pt1) > 2:
            local_disp[i-1, 2] = pt2[2] - pt1[2]
          else:
            local_disp[i-1, 2] = 0
          local_disp[i-1, 3] = np.linalg.norm(disp2)

      pt1 = pt2

  pt0 = l_pt[0]
  pt1 = l_pt[-1]

  disp = pt1 - pt0
  disp2 = np.dot(MP, disp)
  whole_disp[0] = disp2[0]
  whole_disp[1] = disp2[1]

  if len(pt1) > 2:
    whole_disp[2] = pt1[2] - pt0[2]
  else:
    whole_disp[2] = 0
  whole_disp[3] = np.linalg.norm(disp2)
  return local_disp, whole_disp


def build_distance_matrix(points):

    n_bend = len(points)
    nb = 0
    for elt in points:
        if len(elt) > nb:
            nb = len(elt)

    D = np.inf*np.ones((n_bend-1, nb, nb))
    for i in range(1, n_bend):
        for j in range(nb):
            for k in range(nb):
                if i > 0:
                    try:
                        if points[i][j] and points[i-1][k]:
                            d = distance(points[i][j], points[i-1][k])
                        else:
                            d = np.inf
                    except IndexError:
                        continue
                    else:
                        D[i-1][j][k] = d
    return D

def smooth_path(X, Y, x):
    params = np.polyfit(X, Y, deg=2)
    y_smooth = np.polyval(params, x)
    return y_smooth


def compute_sinuosity(data):
    return abs(data[-1][0] - data[0][0]) / distance(data[-1][1:3], data[0][1:3])


def compute_amplitude(pt1, apex, pt3, kind = 'middle'):
    if kind == 'perpendicular':
        pt = project_perpendicularly(apex, pt1, pt3)
    else:
        pt = compute_colinear(pt1, pt3, 0.5)

    amplitude = distance(pt, apex)
    return round(amplitude, 4)

def project_perpendicularly(pt, line_pt1, line_pt2):
    k = (((line_pt2[0] - line_pt1[0]) * (pt[0] - line_pt1[0]) +
          (line_pt2[1] - line_pt1[1]) * (pt[1] - line_pt1[1]))
          / ((line_pt2[0] - line_pt1[0])**2 + (line_pt2[1] - line_pt1[1])**2))
    return compute_colinear(line_pt1, line_pt2, k)


def compute_Leopold_parameters(dataset, meand1, meand2, meand3):

    pt_apex1 = (dataset[meand1[0]]["Cart_abscissa"][meand1[1]], dataset[meand1[0]]["Cart_ordinate"][meand1[1]])
    pt_apex2 = (dataset[meand2[0]]["Cart_abscissa"][meand2[1]], dataset[meand2[0]]["Cart_ordinate"][meand2[1]])
    pt_apex3 = (dataset[meand3[0]]["Cart_abscissa"][meand3[1]], dataset[meand3[0]]["Cart_ordinate"][meand3[1]])

    k = (((pt_apex3[0] - pt_apex1[0]) * (pt_apex2[0] - pt_apex1[0]) +
          (pt_apex3[1] - pt_apex1[1]) * (pt_apex2[1] - pt_apex1[1])) /
         ((pt_apex3[0] - pt_apex1[0])**2 + (pt_apex3[1] - pt_apex1[1])**2))

    pt = compute_colinear(pt_apex1, pt_apex3, k)

    ampl = distance(pt, pt_apex2)
    wavelength = distance(pt_apex1, pt_apex3)

    return wavelength, ampl


def compute_curvature(pt1, pt2, pt3):
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    ds12 = distance(pt1, pt2)
    ds23 = distance(pt2, pt3)
    ds13 = ds12 + ds23

    dxds = (x3 - x1) / (ds13)
    dyds = (y3 - y1) / (ds13)

    d2xds2 = 2 * ( ds12*(x3-x2) - ds23*(x2-x1) ) / (ds12*ds23*ds13)
    d2yds2 = 2 * ( ds12*(y3-y2) - ds23*(y2-y1) ) / (ds12*ds23*ds13)

    curv2 = -(dxds*d2yds2 - dyds*d2xds2) / pow( pow(dxds, 2) + pow(dyds, 2) , 3./2.)

    return curv2

def resample_centerline(x, y, nb_pts=False):
    tck, u = splprep([x, y], s=0)
    if nb_pts:
        u = np.linspace(0., 1., nb_pts)
    return splev(u, tck)


def smooth_centerline(array, window):
    return savgol_filter(array, window, polyorder=3)


def sort_key(labels, reverse=False):
    labels_int = [eval(val) for val in labels]
    labels_int.sort(reverse=reverse)
    labels2 = [str(val) for val in labels_int]
    return labels2

def barycenter(l_val, l_pond):

    if len(l_val) != len(l_pond):
        print("Error: the length of the lists of values and ponderators must be the same to compute the barycenter")
        return 0
    mean = 0
    for val, pond in zip(l_val, l_pond):
        mean += val * pond

    return mean / sum(l_pond)


def get_keys_from_to(all_keys, key_min = 0, key_max = 999999, sort_reverse=False):
    lkeys = []
    for key in all_keys:
        if int(key) <= int(key_max) and int(key) >= int(key_min):
            lkeys += [key]

    if len(lkeys) > 1:
        lkeys = sort_key(lkeys, sort_reverse)

    return [str(key) for key in lkeys]

