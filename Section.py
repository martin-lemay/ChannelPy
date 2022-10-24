# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:41:08 2019

@author: Martin Lemay

Class Section
"""

import numpy as np
import centerline_process_function as cpf

class Section:
    """ Cross-section object to store 2D stratigraphy
    Params: - section id
            - bend id crossed by the section
            - start point coordinates [numpy 1D array]
            - end point coordinates [numpy 1D array]
            - list of Isoline object
            - list of boolean (size of isolines)
            - list of Cl_point object
            - flow direction [numpy 1D array]
    """

    def __init__(self, ide, bend_id, pt_start, pt_stop, isolines, same_bend=None,
                 cl_pts = (), flow_dir=np.array([1,0])):

        self.id = ide
        self.bend_id = bend_id
        self.pt_start = pt_start
        self.pt_stop = pt_stop
        self.dir = np.array(pt_start)-np.array(pt_stop)
        self.dir /= np.linalg.norm(self.dir)

        self.isolines = np.array(isolines)
        self.isolines_origin = self.compute_origin(flow_dir)
        self.cl_pts = cl_pts
        self.same_bend = same_bend
        if same_bend is None:
            self.same_bend = [True for i in range(len(isolines))]
        self.local_disp = None
        self.averaged_disp = None
        self.stacking_pattern_type = None

    def compute_origin(self, flow_dir=np.array([1,0])):
        isolines_origin = []

        # use the orthogonal vector to the flow dir to find the sign
        flow_dir_perp = cpf.perp(flow_dir)

        cl_pt_ref = self.isolines[0].cl_pt_ref
        for i, isoline in enumerate(self.isolines):
            cl_pt = isoline.cl_pt_ref
            # direction of migration according to the flow direction
            sign = 1
            # normed apparent mig vector
            vec = cl_pt.pt[:2] - cl_pt_ref.pt[:2]
            norm_vec = np.linalg.norm(vec)

            if norm_vec > 0:
                vec /= norm_vec
                # scalar product
                dot = np.dot(flow_dir_perp, vec)
            else:
                dot = 1

            if dot < 0:
                sign = -1

            d = sign * cpf.distance(cl_pt_ref.pt, cl_pt.pt) # distance to cl_pt_ref
            dz = cl_pt_ref.pt[2] - cl_pt.pt[2]

            isolines_origin += [(d, dz)]
        return isolines_origin

    # return stacking pattern type from Lemay et al. (2023, GSL)
    def get_stacking_pattern_type(self, mig_threshold, frac_threshold = 0.95, begin_threshold = 0.1):
        mig_steps = []
        pt_origin_prev = (0,0)
        for i, pt_origin in enumerate(self.isolines_origin):
            if i == 0:
                continue

            mig = pt_origin[0] - pt_origin_prev[0]
            if abs(mig) < mig_threshold:
                mig_steps += [0]
            else:
                if mig > 0:
                    mig_steps += [1]
                else:
                    mig_steps += [-1]

            pt_origin_prev = pt_origin

        mig_steps = np.array(mig_steps)

        frac_0 = np.sum(mig_steps == 0) / mig_steps.size
        frac_1 = np.sum(mig_steps > 0)  / mig_steps.size
        frac_2 = np.sum(mig_steps < 0)  / mig_steps.size

        if ((frac_1 > frac_threshold) | (frac_2 > frac_threshold)):
            # print("%s: 1 way migration"%(self.id))
            self.stacking_pattern_type = 0

        elif (((frac_1+frac_0) > frac_threshold) | ((frac_2+frac_0) > frac_threshold)):
            groups = []
            types = []
            prev_mig_step = 2
            for mig_step in mig_steps:
                if ((frac_1>frac_2) & (mig_step == -1)):
                    continue
                if ((frac_1<frac_2) & (mig_step == 1)):
                    continue
                if mig_step == prev_mig_step:
                    groups[-1] += 1
                else:
                    groups += [1]
                    types += [mig_step]
                prev_mig_step = mig_step


            index0 = 1 # index of the first phase of aggradation
            index1 = 0 # index of the first phase of migration
            if 0 in types:
              index0 = types.index(0)
            if 1 in types:
              index1 = types.index(1)

            if ((index0 == 1) & (groups[index1] > begin_threshold*mig_steps.size)):
              # print("%s: 1 way migration"%(self.id))
              self.stacking_pattern_type = 0
            elif (groups[index0] > begin_threshold*mig_steps.size):
                # print("%s: Aggradation + 1 way migration"%(self.id))
                self.stacking_pattern_type = 1
            else:
                # print("%s: 1 way migration"%(self.id))
                self.stacking_pattern_type = 0
        else:
            groups = []
            types = []
            prev_mig_step = 2
            for mig_step in mig_steps:
                if mig_step == 0:
                    continue
                if mig_step == prev_mig_step:
                    groups[-1] += 1
                else:
                    groups += [1]
                    types += [mig_step]
                prev_mig_step = mig_step

            groups = list(filter(lambda a: a != 1, groups))
            if (len(groups) == 1):
                self.stacking_pattern_type = 0 # should not happen
                # print(self.id)
            elif (len(groups) == 2):
                # print("%s: 2 ways migration"%(self.id))
                self.stacking_pattern_type = 2
            else:
                # print("%s: Multi ways migration"%(self.id))
                self.stacking_pattern_type = 3

        return self.stacking_pattern_type

    def channel_apparent_displacements(self, norm_hor=1, norm_vert=1,
                                       write_results=False, filepath="",
                                       smooth=False):

        l_pt = [pt_origin for pt_origin in self.isolines_origin]
        # smooth isolines loc
        if smooth:
            ages = [isoline.age for isoline in self.isolines]
            l_pt = cpf.smooth_trajec(l_pt, ages, ages, 2, resample_curve=False)
        self.local_disp = np.full((len(l_pt)-1,3), np.nan)
        pt_origin_prev = (0,0)
        for i, pt_origin in enumerate(l_pt):
            if i == 0:
                continue
            self.local_disp[i-1, 0] = pt_origin[0] - pt_origin_prev[0] # lateral displacements
            self.local_disp[i-1, 1] = pt_origin[1] - pt_origin_prev[1] # vertical displacements
            pt_origin_prev = pt_origin
        self.local_disp[:, 2] = (self.local_disp[:,0] / self.local_disp[:,1]) * (norm_vert / norm_hor)


    def section_averaged_channel_displacements(self, norm_hor=1, norm_vert=1, mig_threshold=0.1,
                                       write_results=False, filepath=""):
        self.averaged_disp = {}
        self.averaged_disp["full"] = self.compute_average_disp(self.isolines, norm_hor, norm_vert, mig_threshold, True)
        self.averaged_disp["bend"] = self.compute_average_disp(self.isolines, norm_hor, norm_vert, mig_threshold, False)

        if write_results:
            with open(filepath, 'a') as fout:
                fout.write("%s;%s;%s;%s;"%self.averaged_disp["full"][2:])
                fout.write("%s;%s;%s;%s\n"%self.averaged_disp["bend"][2:])


    def compute_average_disp(self, isolines, width, depth, mig_threshold=0.1, whole_trajec=True):
        Dx, Dz, Bcb, Hcb, Bcb_norm, Hcb_norm, Bcb_on_Hcb, Msb = np.zeros(8)
        pt_apex = self.isolines_origin[-1]
        pt_ref = self.isolines_origin[0]
        if not whole_trajec:
            if (self.stacking_pattern_type is None) or ((self.stacking_pattern_type is not None) & (self.stacking_pattern_type > 0)):
                pt_ref = pt_apex
                dmax = 0
                cpt = 0
                for pt_origin in self.isolines_origin[::-1]:
                    d = abs(pt_apex[0] - pt_origin[0])
                    if d > (dmax):
                        dmax = d
                        pt_ref = pt_origin
                        cpt = 0
                    else:
                      cpt += 1

                    if cpt > 3:
                      break

        Dx = round(abs(pt_apex[0] - pt_ref[0]), 4)
        Dz = round(abs(pt_apex[1] - pt_ref[1]), 4)

        if Dz != 0:
            Msb = round((Dx / Dz) * (depth / width), 4)
            Hcb = Dz + depth # full channel belt thickness
            Bcb = Dx + width # full channel belt width
            Bcb_on_Hcb = round(Bcb / Hcb, 4)
        else:
            Msb = -99999
            Bcb_on_Hcb = -99999
            if Dz == 0:
                Hcb = -99999
            else:
                Hcb = Dz + depth

            if Dx == 0:
                Bcb = -99999
            else:
                Bcb = Dx + width

        Bcb_norm = round(Bcb / width, 4)
        Hcb_norm = round(Hcb / depth, 4)

        return np.array([Dx, Dz, Bcb, Hcb, Bcb_norm, Hcb_norm, Bcb_on_Hcb, Msb])
