import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

class Cl_point:
    """ Centerline point. Store coordinates and all the associated variables """

    def __init__(self, dataset):

        self.s = dataset["Curv_abscissa"]
        self.pt = (dataset["Cart_abscissa"], dataset["Cart_ordinate"], dataset["Elevation"])
        self.data = dataset # curvature, height, velocity, ...
    
    def __repr__(self):
        return self.pt

    # add the properties of self with those of cl_point
    # return a new Cl_point
    def __add__(self, cl_point):
        array = [self.data[col] + cl_point.data[col] for col in self.data.index]
        data = pd.Series(array, index=self.data.index)
        return Cl_point(data)
    
    # multiply the properties of self by a scalar n
    # return a new Cl_point
    def __mul__(self, n):
        array = [n * self.data[col] for col in self.data.index]
        data = pd.Series(array, index=self.data.index)
        return Cl_point(data)                  
                
    def set_curvature(self, curv):
        self.data["Curvature"] = curv
    
    def curvature(self):
        return self.data["Curvature"]
    
    def velocity(self):
        return self.data["Velocity"]

    def depth(self):
        return self.data["Depth_mean"]   
        
class Bend:
    """ Store and compute bend parameters from Centerline object
        Params: - bend id
                - centerline bewteen inflection points
                - index of the upstream inflection point along the centerline
                - index of the downstream inflection point along the centerline
                - number of points among which the apex is choosen
    """

    def __init__(self, bend_id, centerline, inflex_up, inflex_down, age=0, 
                 nb=1, sinuo_thres = 1):   
        self.id = bend_id
        self.age = age
        self.isvalid = False # valid if the sinuosity higher than the threshold
        self.index_inflex_up = inflex_up
        self.index_inflex_down = inflex_down
        
        self.cl_points = centerline
        self.cl_pt_inflex_up = self.cl_points[0] # upstream inflection point
        self.cl_pt_inflex_down = self.cl_points[-1] # downstream inflection point

        self.pt_middle = compute_colinear(self.cl_pt_inflex_up.pt, self.cl_pt_inflex_down.pt, 0.5)

        # Sinuosity, Length, half-wavelength, Amplitude perpendicular, Amplitude middle
        # individual meander geometry
        self.params = pd.Series(np.nan*np.zeros(7), index=("Sinuosity", "Length", "Half_Wavelength",
                                                           "Amplitude_perp", "Amplitude_middle",
                                                           "Amplitude_Leopold", "Wavelength_Leopold"))
        
        # meander geometry averaged over a given window (computed later)
        self.params_averaged = pd.DataFrame(np.nan*np.zeros((1,7)),
                                            columns=self.params.index)

        # find the apex of the bend and compute bend geometric properties (self.params)
        self.side = False
        self.cl_pt_max_curv = self.find_apex(nb, sinuo_thres)
        self.cl_pt_max_curv # may be changed later if several consecutive centerlines

        # compute the amplitudes
        self.params["Amplitude_perp"] = self.compute_amplitude(self.cl_pt_apex.pt, kind="perpendicular")
        self.params["Amplitude_middle"] = self.compute_amplitude(self.cl_pt_apex.pt, kind="middle")
    
    
    def __repr__(self):
        return str(self.id)
    
    
    # add properties of self and another bend
    # return a new bend with the same id as self
    def __add__(self, bend, nb=1, sinuo_thres=1):
        
        new_bend = Bend(self.id, self.cl_points + bend.cl_points, 
                       self.cl_pt_inflex_up, bend.cl_pt_inflex_down, 
                       self.age, nb=nb, sinuo_thres=sinuo_thres)
        return new_bend
    
    
    def bends(self, bend_id):
        i = 0
        ret = False
        while not i and i < len(bends):
            if bends[i] == bend_id:
                ret = i
            i += 1            
        return ret
        
        
    # apex = maximal curvature among the nb farest points (euclidienne distance) from the middle of the segment between inflexion points (or perpendicular to this segment for low amplitude meander (< 0.5*d_inflex))
    # if nb = 1, take the point at maximal amplitude
    def find_apex2(self, nb=1, sinuo_thres = 1):

        self.params["Length"] = abs(self.cl_pt_inflex_down.s - self.cl_pt_inflex_up.s)

        d_inflex = distance(self.cl_pt_inflex_up.pt, self.cl_pt_inflex_down.pt)
        self.params["Half_Wavelength"] = d_inflex
        if d_inflex > 0 and d_inflex < self.params["Length"]:
            self.params["Sinuosity"] = self.params["Length"] / d_inflex
        else:
            self.params["Sinuosity"] = 1

        if self.params["Sinuosity"] >= sinuo_thres:
            self.isvalid = True

        # compute the amplitude of each centerline points
        ampl = [self.compute_amplitude(cl_pt.pt, 'middle') for cl_pt in self.cl_points]
        # if low amplitude meander, we search for the greatest perpendicular distance
        if (max(ampl) <= 0.5 * d_inflex+0.01):
            ampl = [self.compute_amplitude(cl_pt.pt, 'perpendicular') for cl_pt in self.cl_points]

        # keep the nb points with the greatest amplitude
        l_index = []
        if len(ampl) <= nb:
            l_index = [i for i in range(len(ampl))]
        else:
            ampl2 = [val for val in ampl]
            for i in range(nb):
                l_index += [ampl.index(max(ampl2))]
                ampl2.remove(max(ampl2))

        # we keep the one with the maximum curvature
        curv = 0
        for i in l_index:
            if abs(self.cl_points[i].curvature()) > curv:
                curv = abs(self.cl_points[i].curvature())
                index = i

        self.index_apex = index
        self.cl_pt_apex = self.cl_points[index]
        if self.cl_pt_apex.curvature()>0:
            self.side = "up"
        else:
            self.side = "down"
        
        return True


    # apex = maximal amplitude among the nb highest curvature points
    def find_apex(self, nb=1, sinuo_thres=1):

        self.params["Length"] = abs(self.cl_pt_inflex_down.s - self.cl_pt_inflex_up.s)

        d_inflex = distance(self.cl_pt_inflex_up.pt, self.cl_pt_inflex_down.pt)
        self.params["Half_Wavelength"] = d_inflex
        if d_inflex > 0 and d_inflex < self.params["Length"]:
            self.params["Sinuosity"] = self.params["Length"] / d_inflex
        else:
            self.params["Sinuosity"] = 1

        if self.params["Sinuosity"] >= sinuo_thres:
            self.isvalid = True

        # we keep the nb points with the maximum curvature
        # look for the nb points with the highest curvature
        curv = list(np.linspace(-1,0,nb))
        indexes = list(np.zeros(nb))
        for index, cl_pt in enumerate(self.cl_points):
            if abs(cl_pt.curvature()) > min(curv):
                k = curv.index(min(curv))
                curv[k] = abs(cl_pt.curvature())
                indexes[k] = index

        # compute amplitudes
        ampl = [self.compute_amplitude(self.cl_points[int(i)].pt, 'middle') for i in indexes]
        # if low amplitude meander, we search for the greatest perpendicular distance
        if (max(ampl) <= 0.5 * d_inflex+0.01):
            ampl = [self.compute_amplitude(self.cl_points[int(i)].pt, 'perpendicular') for i in indexes]

        # keep the point with the greatest amplitude
        index = indexes[ampl.index(max(ampl))]

        self.index_apex = index
        self.cl_pt_apex = self.cl_points[index]

        return True


    def compute_amplitude(self, pt, kind = 'middle'):

        # use the perpendicular projection of the apex on the line defined by inflectin points
        if kind == 'perpendicular':
            k = (((self.cl_pt_inflex_down.pt[0] - self.cl_pt_inflex_up.pt[0]) *
                  (pt[0] - self.cl_pt_inflex_up.pt[0]) +
                  (self.cl_pt_inflex_down.pt[1] - self.cl_pt_inflex_up.pt[1]) *
                  (pt[1] - self.cl_pt_inflex_up.pt[1])))
            den = ((self.cl_pt_inflex_down.pt[0] - self.cl_pt_inflex_up.pt[0])**2 +
                  (self.cl_pt_inflex_down.pt[1] - self.cl_pt_inflex_up.pt[1])**2)

            if den != 0:
                k /= den
            else:
                k = 0 # the 2 points are superposed
        # use the middle between inflection points
        else:
            k = 0.5
        pt_proj = compute_colinear(self.cl_pt_inflex_up.pt, self.cl_pt_inflex_down.pt, k)
        amplitude = distance(pt_proj, pt)

        return round(amplitude, 4)


class Centerline:
    """ Centerline object, collection of Cl_point """

    def __init__(self, age, dataset, spacing, smooth_distance, lag=1, nb=1, sinuo_thres=1,
                 compute_curvature=True, interpol_props=True, plot_curvature=False, compute_geometry=True):

        self.age = age
        self.cl_points = []

        self.init_centerline(dataset, spacing, smooth_distance, compute_curvature, interpol_props,
                             plot_curvature=plot_curvature, compute_geometry=compute_geometry)
        self.bends = []
        self.bends_filtered = []
        
        self.find_bends(lag, nb, sinuo_thres)
        self.filter_bends(nb, sinuo_thres)
        
        if compute_geometry:
            self.compute_geometry_leopold()

    def init_centerline(self, dataset, spacing, smooth_distance, compute_curvature=True,
                        interpol_props=False, plot_curvature=False, compute_geometry=True):

        # 1. resample the centerline with a parametric spline function
        nb_pts = int(self.compute_total_length(dataset["Cart_abscissa"], dataset["Cart_ordinate"]) / spacing +1)
        
        if nb_pts > dataset.shape[0]:
            new_points = resample_centerline(dataset["Cart_abscissa"], dataset["Cart_ordinate"], nb_pts)

            dataset_new = pd.DataFrame(np.zeros((len(new_points[0]), len(dataset.columns))),
                                       columns=dataset.columns)

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
                props = [name for name in dataset.columns if name not in ("Curv_abscissa", "Cart_abscissa", "Cart_ordinate")]
                self.interpol_properties(dataset_new, dataset, props)

            # 3. compute and smooth curvatures
            if compute_curvature:
                self.compute_curvature(dataset_new, window)
        else:
            dataset_new = dataset
            print("WARNING: with input spacing, total number of points lower than input data")
            
        # Create Centerline object as a collection of cl_Points
        for i, row in dataset_new.iterrows():
            self.cl_points += [Cl_point(row)]

        if plot_curvature:
            plt.figure()
            plt.plot(dataset_new["Curv_abscissa"], dataset_new["Curvature"], 'k-')
            plt.plot([0, dataset_new["Curv_abscissa"].tolist()[-1]], [0, 0], '--', color='grey')
            plt.show()

    def compute_curvilinear_abscissa(self, dataset):

        ls = np.zeros(dataset["Cart_abscissa"].size)
        pt, pt_prev = (0,0), (0,0)
        for i, row in dataset.iterrows():
            pt = (row["Cart_abscissa"], row["Cart_ordinate"])
            if i > 0:
                ls[i] = ls[i-1] + distance(pt, pt_prev)
            pt_prev = pt

        dataset["Curv_abscissa"] = ls

    def compute_curvature(self, dataset, window):

        lcurv = np.zeros(len(dataset["Cart_abscissa"]))
        for i, row in dataset.iterrows():

            if i > 0 and i < len(lcurv)-1:

                x1, y1 = dataset["Cart_abscissa"][i-1], dataset["Cart_ordinate"][i-1]
                x2, y2 = row["Cart_abscissa"], row["Cart_ordinate"]
                x3, y3 = dataset["Cart_abscissa"][i+1], dataset["Cart_ordinate"][i+1]

                pt1 = (x1, y1)
                pt2 = (x2, y2)
                pt3 = (x3, y3)

                ds12 = distance(pt1, pt2)
                ds23 = distance(pt2, pt3)
                ds13 = ds12 + ds23

                dxds = (x3 - x1) / (ds13)
                dyds = (y3 - y1) / (ds13)

                d2xds2 = 2 * ( ds12*(x3-x2) - ds23*(x2-x1) ) / (ds12*ds23*ds13)
                d2yds2 = 2 * ( ds12*(y3-y2) - ds23*(y2-y1) ) / (ds12*ds23*ds13)

                dataset["Curvature"][i] = -(dxds*d2yds2 - dyds*d2xds2) / pow( pow(dxds, 2) + pow(dyds, 2) , 3./2.)

        # smooth curvature using the Savitzky-Golay filter
        if window % 2==0:
            window += 1 # to be odd
        if window <= 3:
            print("WARNING: curvature smoothing window is 5")
            window = 5

        dataset["Curvature"] = savgol_filter(dataset["Curvature"], window, polyorder=3)

    def compute_total_length(self, X, Y):
        s = 0
        x_prev, y_prev = 0, 0
        for i, x in enumerate(X):
            y = Y[i]
            if i > 0:
                s += distance((x, y), (x_prev, y_prev))
            x_prev, y_prev = x, y
        print("total length", s)
        return s

    def resample_centerline(self, x, y, nb_pts=False):
        tck, u = splprep([x, y], s=0)
        if nb_pts:
            u = np.linspace(0., 1., nb_pts)
        return splev(u, tck)

    def find_closest_point(self, pt, dataset, index=0):
        """ Find the point from the old centerline the closest to the point x,y
            on the new resampled centerline between index and index+1
            Return the index of the closest point in dataset
        """

        d1 = 0
        d2 = 0
        #index = dataset.shape[0]-1        
        if index < dataset.shape[0]-1:
            d1 = distance((dataset.loc[index, "Cart_abscissa"], 
                               dataset.loc[index, "Cart_ordinate"]), 
                               pt)
            d2 = distance((dataset.loc[index+1, "Cart_abscissa"], 
                               dataset.loc[index+1, "Cart_ordinate"]), 
                               pt)          
        if d1 <= d2:
            return index
        else:
            return index+1
        
    def interpol_properties(self, dataset_new, dataset, props):
               
        j1 = 0 # index of the closest point in dataset
        for i, row in dataset_new.iterrows():

            if i == 0 or i == dataset_new.shape[0]-1:
                # copy the properties of the first and last points
                d1 = 0
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
                pt_proj, j2 = project_point(pt_new0, pt_new1, pt_new2, pt0, pt1, pt2)
                
                # 3. interpolate the properties - compute the distances
                d1 = distance(pt_proj, pt1)
                if j2<0:
                    d2 = distance(pt_proj, pt0)
                elif j2>0:
                    d2 = distance(pt_proj, pt2)
                else:
                    d2 = 0
                    
            # 3. interpolate the properties - compute them into the new point    
            if d1 > 0:
                denom = d1 + d2
                for prop in props:
                    dataset_new.loc[i, prop] = (d1 * dataset[prop][j1] +
                                                d2 * dataset[prop][j1+j2]) / denom
            else:
                for prop in props:
                    dataset_new.loc[i, prop] = dataset[prop][j1]
        
        return True              

    def find_inflexion_points(self, lag=1):
        """ Find all inflection points from the object Centerline
            Return the list of inflection point indexes
        """
        inflex_pts = [-(2+lag)]
        prev_curv = 0
        for i, cl_pt in enumerate(self.cl_points):
            curv = cl_pt.curvature()

            if (i > 0 and curv * prev_curv < 0):

                if (i > inflex_pts[-1]+lag):
                    inflex_pts += [i]
                else:
                    inflex_pts.remove(inflex_pts[-1])

            prev_curv = curv
        inflex_pts.remove(-(2+lag))
        return inflex_pts

    def find_bends(self, lag=1, nb=1, sinuo_thres=1):

        inflex_pts = self.find_inflexion_points(lag)
        prev_inflex = 0
        for i, inflex in enumerate(inflex_pts):
            bend = Bend(i, self.cl_points[prev_inflex:inflex+1], prev_inflex,
                             inflex, self.age, nb=nb, sinuo_thres=sinuo_thres)
            self.bends += [bend]
            prev_inflex = inflex

        # add last bend part
        self.bends += [Bend(i+1, self.cl_points[prev_inflex:len(self.cl_points)], prev_inflex,
                             inflex, nb=nb, sinuo_thres=sinuo_thres)]
        return True

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
                    for j in range(1, k+1, 1):
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
                        
                # if k is odd, means that the 2 consucutive valid bends are not by the same side    
                elif k%2==0:                       
                    # get the middle
                    best_s = (self.bends_filtered[-1].cl_pt_apex.s + self.bends[i+k].cl_pt_apex.s)/2
                    # get the inflexion point the closest from the middle 
                    ls_inflex = np.array([best_s - self.bends[i+j].cl_pt_inflex_up.s for j in range(k)])
                    n = ls_inflex.argmin()
                    
                    # gather the last valid bend until those until the middle
                    for j in range(1,n):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]
                    # gather the bend in the middle the next ones until the next valid one included
                    self.bends_filtered += [self.bends[i+n]]
                    for j in range(n+1,k+1):
                        self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i+j]
                                   
        return True

    def compute_geometry_leopold(self):

        prev_bend = False
        bend = False
        for i, next_bend in enumerate(self.bends):
            if i > 1:
                bend.params["Wavelength_Leopold"] = distance(prev_bend.cl_pt_apex.pt, next_bend.cl_pt_apex.pt)
                pt_proj = project_perpendicularly(bend.cl_pt_apex.pt, prev_bend.cl_pt_apex.pt, next_bend.cl_pt_apex.pt)
                bend.params["Amplitude_Leopold"] = distance(bend.cl_pt_apex.pt, pt_proj)

            prev_bend = bend
            bend = next_bend

    def morphometry(self, window_size, props = ("Half_Wavelength", "Amplitude_perp", "Amplitude_middle",
                 "Wavelength_Leopold", "Amplitude_Leopold", "Length")):

        mean_values = pd.Series(np.zeros(self.bends[0].params.size), index=self.bends[0].params.index)
        for i, bend in enumerate(self.bends):

            smin = bend.cl_points[0].s - window_size
            smax = bend.cl_points[0].s + window_size

            jmin = i
            ptmin = bend.cl_points[0]
            while (self.bends[jmin].cl_points[0].s > smin and jmin>0):
                jmin -= 1
                ptmin = self.bends[jmin].cl_points[0]

            jmax = i
            ptmax = bend.cl_points[-1]
            while (self.bends[jmax].cl_points[-1].s < smax and jmax < len(self.bends)-1):
                jmax += 1
                ptmax = self.bends[jmax].cl_points[-1]

            bend.params_averaged["Length"][0] = abs(ptmax.s-ptmin.s)
            d = distance(ptmin.pt, ptmax.pt)

            if d > 0 and d < bend.params_averaged["Length"][0]:
                bend.params_averaged["Sinuosity"][0] = bend.params_averaged["Length"][0] / d
            else:
                bend.params_averaged["Sinuosity"][0] = 1 
            nb = 0
            for j in np.arange(jmin, jmax+1):
                nb += 1
                for prop in props:
                    if j == jmin:
                        bend.params_averaged[prop][0] = 0
                    if np.isfinite(self.bends[j].params[prop]):
                        bend.params_averaged[prop][0] += self.bends[j].params[prop]

            for prop in props:
                bend.params_averaged[prop][0] /= nb

                if np.isfinite(bend.params[prop]):
                    mean_values[prop] += bend.params[prop]

        mean_values /= self.bends[-1].id
        mean_values["Sinuosity"] = (abs(self.cl_points[-1].s-self.cl_points[0].s) / 
                                       distance(self.cl_points[0].pt, self.cl_points[-1].pt))
        return mean_values

    def save_morphometry_results(self, output_path, delimiter=';'):

        props = ["Sinuosity",
                 "Wavelength_Leopold",
                 "Amplitude_Leopold",
                 "Half_Wavelength",
                 "Amplitude_middle"]

        data = pd.DataFrame(np.nan*np.zeros((len(self.bends), len(props)+1)),
                            columns = ["Bend_ID"]+props)

        for i, bend in enumerate(self.bends):
            data["Bend_ID"][i] = bend.id
            for prop in props:
                data[prop][i] = bend.params_averaged[prop][0]

        data.to_csv(output_path, sep=delimiter, index=False, float_format='%.2f')


def distance(pt1, pt2):
    d = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    return round(d, 4)

def norm(vec):    
    return np.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
 
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

def compute_colinear(pt1, pt2, k):
    x = pt1[0] + k * (pt2[0] - pt1[0])
    y = pt1[1] + k * (pt2[1] - pt1[1])
    return (x, y)

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

def project_perpendicularly(pt, line_pt1, line_pt2):
    k = (((line_pt2[0] - line_pt1[0]) * (pt[0] - line_pt1[0]) +
          (line_pt2[1] - line_pt1[1]) * (pt[1] - line_pt1[1]))
          / ((line_pt2[0] - line_pt1[0])**2 + (line_pt2[1] - line_pt1[1])**2))
    return compute_colinear(line_pt1, line_pt2, k)

def resample_centerline(x, y, nb_pts=False):
    tck, u = splprep([x, y], s=0)
    if nb_pts:
        u = np.linspace(0., 1., nb_pts)
    return splev(u, tck)

# create the dataset from input file
def create_dataset_from_xyz(X, Y, Z):
    data = np.zeros((X.size, 5))
    s = 0
    x_prev, y_prev = 0, 0
    for i, x in enumerate(X):
        y = Y[i]
        z = Z[i]
        data[i, 1] = x
        data[i, 2] = y
        data[i, 3] = z
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

    # first and last curvatures
    data[0, 4] = data[1, 4]
    data[-1, 4] = data[-2, 4]
    
    dataset = pd.DataFrame(data, columns=("Curv_abscissa", "Cart_abscissa", "Cart_ordinate",
                                                  "Elevation", "Curvature"))
    return dataset
    

def plot_bends(ax, bends, domain = [[],[]], annotate = False,
               plot_apex = True, plot_inflex = False, plot_middle = False,
               plot_apex_trajec = True,
               annot_text_size=10, color_bend = False, alpha=1,
               linewidth=1, markersize=2, cl_color=False):
    
    if color_bend and not cl_color:
        colors = ('b', 'r')
    else:
        colors = (cl_color, cl_color)
        
    for i, bend in enumerate(bends):
        abscissa = []
        ordinates = []
        for cl_pt in bend.cl_points:
            abscissa += [cl_pt.pt[0]]
            ordinates += [cl_pt.pt[1]]

        ax.plot(abscissa, ordinates, linestyle='-', linewidth=linewidth, color=colors[i%2], alpha=alpha)

        if plot_inflex:
            ax.plot(bend.cl_pt_inflex_up.pt[0], bend.cl_pt_inflex_up.pt[1],
                    marker = 'o', markerfacecolor='green', markeredgecolor='k',
                    markersize = markersize)
            if i == len(bends)-1:
                ax.plot(bend.cl_pt_inflex_down.pt[0], bend.cl_pt_inflex_down.pt[1],
                        marker = 'o', markerfacecolor='green', markeredgecolor='k',
                        markersize = markersize)

        if plot_apex and bend.isvalid:
            ax.plot(bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1],
                    marker = 'd', markeredgecolor='k', markerfacecolor='r', 
                    markersize = 1.5*markersize)


        if plot_middle and bend.isvalid:
            ax.plot(bend.pt_middle[0], bend.pt_middle[1],
                    marker = 'o', color='k', markersize = 0.8*markersize)

        if annotate:
            if len(domain[0])>0:
                if (bend.cl_pt_apex.pt[0] > domain[0][0] and bend.cl_pt_apex.pt[0] < domain[0][1] and
                    bend.cl_pt_apex.pt[1] > domain[1][0] and bend.cl_pt_apex.pt[1] < domain[1][1]):
                    plt.text(bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1], str(bend.id), size=10)
            else:
                x,y = bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1]
                plt.text(x, y, str(bend.id), size=annot_text_size,
                         horizontalalignment="center")
