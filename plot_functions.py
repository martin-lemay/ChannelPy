# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:54:46 2019

@author: Martin Lemay

Plot functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import centerline_process_function as cpf


def plot_centerline_collection(work_dir, cl_collec, domain, nb_cl = 999, show = False,
                               annotate = False, plot_apex = True, plot_inflex = False,
                               plot_middle = False, plot_centroid = False,
                               annot_text_size=10, color_bend = False,
                               plot_apex_trajec = True, plot_centroid_trajec = False,
                               plot_normal = False, scale_normal = 1., plot_section = False,
                               plot_warping = True, cmap_name='Blues'):

    # get the centerlines to plot
    if nb_cl == 999:
        keys = cl_collec.all_iter
    else:
        ite = np.linspace(np.min(cl_collec.all_iter), np.max(cl_collec.all_iter), nb_cl)
        keys = np.empty_like(ite)
        for i, it in enumerate(ite):
            diff = np.abs(cl_collec.all_iter - it)
            keys[i] = cl_collec.all_iter[diff==np.min(diff)]


    cmap = cm.get_cmap(cmap_name)
    cmap_norm = colors.Normalize(vmin=keys[0], vmax=keys[-1])

    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    for i, key in enumerate(keys):
        cl = cl_collec.centerlines[key]
        cl_color = cmap(cmap_norm(key))

        if key == np.max(cl_collec.all_iter):
            plot_bends(ax, (cl.cl_points,), cl.bends, domain=domain, annotate = annotate,
                       plot_apex = plot_apex, plot_inflex = plot_inflex, plot_middle = plot_middle,
                       plot_centroid=plot_centroid, plot_normal=plot_normal, scale_normal=scale_normal,
                       annot_text_size=annot_text_size, color_bend=color_bend,
                       alpha=1, linewidth=2, markersize=2, cl_color=cl_color)
        else:
            plot_bends(ax, (cl.cl_points,), cl.bends, domain=domain, annotate = False,
                       plot_apex = plot_apex, plot_inflex = plot_inflex, plot_middle = plot_middle,
                       plot_centroid=plot_centroid, plot_normal=plot_normal, scale_normal=scale_normal,
                       annot_text_size=annot_text_size, color_bend=color_bend,
                       alpha=1, linewidth=1, markersize=1, cl_color=cl_color)

    if plot_apex_trajec:
        for bend_evol in cl_collec.bends_evol:
            coords = cpf.points2coords(bend_evol.apex_trajec_smooth)
            ax.plot(coords[0], coords[1], 'r-', linewidth=1)

    if plot_centroid_trajec:
        for bend_evol in cl_collec.bends_evol:
            coords = cpf.points2coords(bend_evol.centroid_trajec_smooth)
            ax.plot(coords[0], coords[1], '-', color='orange', linewidth=1)

    if plot_section and cl_collec.section_lines:
        for section_line in cl_collec.section_lines:
            coords = np.array(section_line)
            ax.plot((coords[0][0], coords[1][0]),
                    (coords[0][1], coords[1][1]),
                    'k-', linewidth=1)

    if plot_warping:
      try:
        for i, key2 in enumerate(cl_collec.all_iter[:-1]):

          key1 = cl_collec.all_iter[i+1]

          ctls1 = cl_collec.centerlines[key1]
          ctls2 = cl_collec.centerlines[key2]

          if i==0:
            continue

          warp_x, warp_y = [], []
          for index1, index2 in enumerate(ctls1.index_cl_pts_prev_centerline):
            if not np.isfinite(index1) or not np.isfinite(index2):
              continue
            pt1 = ctls1.cl_points[int(index1)].pt
            pt2 = ctls2.cl_points[int(index2)].pt
            warp_x += [[pt1[0], pt2[0]]]
            warp_y += [[pt1[1], pt2[1]]]

          for x, y in zip(warp_x, warp_y):
            plt.plot(x, y, 'k-', linewidth=0.25)
      except:
        print("warping was not plotted")


    if len(domain[0]) > 0:
        plt.xlim(domain[0])
    if len(domain[1]) > 0:
        plt.ylim(domain[1])
    if len(domain[0])+len(domain[1]) == 0:
        plt.axis('equal')

    plt.xticks(np.arange(domain[0][0], domain[0][1]+1, 500), rotation=45)
    plt.yticks(np.arange(domain[1][0], domain[1][1]+1, 500))

    plt.grid(True, which='both', axis='both')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    if work_dir:
        plt.savefig(work_dir + '.png', dpi = 300)
        plt.savefig(work_dir + '.eps', dpi = 300)

    if show:
        plt.show()

    plt.close('all')


def plot_centerline_single(work_dir, cl_points, bends, domain, show = False, annotate = False,
                           plot_apex = True, plot_inflex = False, plot_middle = False,
                           plot_centroid = False, plot_apex_proba = False,
                           plot_normal = True, scale_normal = 1., annot_text_size=10,
                           color_bend=True, linewidth=1, markersize=2, ax=False):

    if not ax:
        fig, ax = plt.subplots(figsize=(5,5), dpi=150)

    plot_bends(ax, cl_points, bends, domain = domain, annotate = annotate,
               plot_apex = plot_apex, plot_inflex = plot_inflex, plot_middle = plot_middle,
               plot_centroid = plot_centroid, plot_apex_proba=plot_apex_proba,
               plot_normal=plot_normal, scale_normal=scale_normal, annot_text_size=annot_text_size,
               color_bend=color_bend, alpha=1, cl_color=False)

    if not ax:
        if len(domain[0]) > 0:
            plt.xlim(domain[0])
        if len(domain[1]) > 0:
            plt.ylim(domain[1])
        if len(domain[0])+len(domain[1]) == 0:
            plt.axis('equal')

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        if work_dir:
            plt.savefig(work_dir + '.png', dpi = 300)
            plt.savefig(work_dir + '.eps', dpi = 300)

        if show:
            plt.show()

        plt.close('all')


def plot_bend_evol(ax, cl_collec, bend_evol, nb_cl = 999, domain = [[],[]], annotate = False,
                    plot_apex = True, plot_inflex = False, plot_middle = False,
                    plot_centroid = False, plot_centroid_trajec = False,
                    plot_apex_trajec = False, plot_middle_trajec = False, plot_section=False,
                    annot_text_size=10, color_bend = False, alpha=1,
                    linewidth=1, markersize=2, cmap_name="Blues"):

    # get the centerlines to plot
    if nb_cl == 999:
        keys = bend_evol.all_iter
    else:
        ite = np.linspace(np.min(bend_evol.all_iter), np.max(bend_evol.all_iter), nb_cl)
        keys = np.empty_like(ite)
        for i, it in enumerate(ite):
            diff = np.abs(bend_evol.all_iter - it)
            keys[i] = bend_evol.all_iter[diff==np.min(diff)]

    cmap = cm.get_cmap(cmap_name)
    cmap_norm = colors.Normalize(vmin=bend_evol.all_iter[-1], vmax=bend_evol.all_iter[0])
    cmap_norm = colors.Normalize(vmin=keys[0], vmax=keys[-1])

    for age, bend_index in bend_evol.bend_indexes.items():
        if age not in keys:
            continue
        cl_color = cmap(cmap_norm(age))

        if age == np.max(keys):
            plot_bends(ax, (cl_collec[0].centerlines[age].cl_points,), [cl_collec[0].centerlines[age].bends[bend_index]],
                       domain = [[],[]], annotate = annotate,
                       plot_apex = plot_apex, plot_inflex = plot_inflex, plot_middle = plot_middle,
                       plot_centroid = plot_centroid, annot_text_size=10, color_bend = color_bend,
                      alpha=1, linewidth=2, markersize=5, cl_color=cl_color)
        else:
            plot_bends(ax, (cl_collec[0].centerlines[age].cl_points,), [cl_collec[0].centerlines[age].bends[bend_index]],
                       domain = [[],[]], annotate = annotate,
                       plot_apex = plot_apex, plot_inflex = plot_inflex, plot_middle = plot_middle,
                       plot_centroid = plot_centroid, annot_text_size=10, color_bend = color_bend,
                       alpha=1, linewidth=1, markersize=2, cl_color=cl_color)




    if plot_apex_trajec and bend_evol.apex_trajec_smooth:
        coords = cpf.points2coords(bend_evol.apex_trajec_smooth)
        ax.plot(coords[0], coords[1], 'r-', linewidth=1)

    if plot_middle_trajec and bend_evol.middle_trajec_smooth:
        coords = cpf.points2coords(bend_evol.middle_trajec_smooth)
        ax.plot(coords[0], coords[1], 'b-', linewidth=1)

    if plot_centroid_trajec and bend_evol.centroid_trajec_smooth:
        coords = cpf.points2coords(bend_evol.centroid_trajec_smooth)
        ax.plot(coords[0], coords[1], '-', color='orange', linewidth=1)

    if plot_section:
        section_indexes = []
        for age, bend_index in bend_evol.bend_indexes.items():
            if (cl_collec[0].centerlines[age].bends[bend_index].intersected_section_indexes):
                section_indexes += cl_collec[0].centerlines[age].bends[bend_index].intersected_section_indexes

        for section in np.unique(section_indexes):
            X, Y = [], []
            for isoline in section.isolines:
                i = len(isoline.points) // 2  # centerline point
                pt = isoline.points[i]
                X += [pt[0]]
                Y += [pt[1]]
                ax.plot(X, Y, 'k-', linewidth=1)


def plot_bends(ax, cl_points, bends, domain = [[],[]], annotate = False,
               plot_apex = True, plot_inflex = False, plot_middle = False,
               plot_centroid = False, plot_centroid_trajec = False, plot_apex_trajec = True,
               plot_apex_proba = False, plot_normal = False, scale_normal = 1.,
               annot_text_size=10, color_bend = False, alpha=1,
               linewidth=1, markersize=2, cl_color=False, plot_vel_perturb=False):

    color = 'k'
    if cl_color:
        color = cl_color

    if plot_vel_perturb:

      vp_colormap = cm.get_cmap("seismic")
      vp_colormap_norm = colors.Normalize(vmin=-1.2, vmax=1.2)

    for i, bend in enumerate(bends):
        abscissa = []
        ordinates = []
        for cl_pt in cl_points[0][bend.index_inflex_up:bend.index_inflex_down+1]:
            abscissa += [cl_pt.pt[0]]
            ordinates += [cl_pt.pt[1]]


        if color_bend:
            color = 'r'
            if bend.side == "up":
                color = 'b'

        ax.plot(abscissa, ordinates, linestyle='-', linewidth=linewidth, color=color, alpha=alpha)

        if plot_inflex:
            ax.plot(cl_points[0][bend.index_inflex_up].pt[0], cl_points[0][bend.index_inflex_up].pt[1],
                    marker = 'o', markerfacecolor='green', markeredgecolor='k',
                    markersize = markersize)
            if i == len(bends)-1:
                ax.plot(cl_points[0][bend.index_inflex_down].pt[0], cl_points[0][bend.index_inflex_down].pt[1],
                        marker = 'o', markerfacecolor='green', markeredgecolor='k',
                        markersize = markersize)

        if plot_apex and bend.isvalid:
            ax.plot(cl_points[0][bend.index_apex].pt[0], cl_points[0][bend.index_apex].pt[1],
                    marker = 'd', markeredgecolor='k', markerfacecolor='r',
                    markersize = 1.5*markersize)


        if plot_middle and bend.isvalid:
            ax.plot(bend.pt_middle[0], bend.pt_middle[1],
                    marker = 'o', color='k', markersize = 0.8*markersize)

        if plot_centroid and bend.isvalid:
            ax.plot(bend.pt_centroid[0], bend.pt_centroid[1],
                    marker = 'o', markeredgecolor='k', markerfacecolor='orange',
                    markersize = 0.8*markersize)
        if plot_apex_proba and bend.isvalid:
            x = []
            y = []
            for i, cl_pt in enumerate(cl_points[0][bend.index_inflex_up:bend.index_inflex_down+1]):
                x += [cl_pt.pt[0]]
                y += [cl_pt.pt[1]]
            ax.scatter(x, y,
                marker = 'o', c=bend.apex_probability, cmap='jet')

        if plot_normal:
            for cl_pt in cl_points[0][bend.index_inflex_up:bend.index_inflex_down+1]:
                plt.arrow(cl_pt.pt[0], cl_pt.pt[1],
                  cl_pt.data["Normal_x"]*scale_normal, cl_pt.data["Normal_y"]*scale_normal,
                  color='k', width=4, linewidth = 1)

        if plot_vel_perturb:
            for cl_pt in cl_points[0][bend.index_inflex_up:bend.index_inflex_down+1]:
                vp_color = vp_colormap(vp_colormap_norm(cl_pt.data["Vel_perturb"]))
                plt.arrow(cl_pt.pt[0], cl_pt.pt[1],
                  cl_pt.data["Normal_x"]*cl_pt.data["Vel_perturb"]*scale_normal, cl_pt.data["Normal_y"]*cl_pt.data["Vel_perturb"]*scale_normal,
                  color=vp_color, width=4, linewidth = 1)

        if annotate & (bend.index_apex):
            cl_pt_apex = cl_points[0][bend.index_apex]
            if len(domain[0])>0:
                if (cl_pt_apex.pt[0] > domain[0][0] and cl_pt_apex.pt[0] < domain[0][1] and
                    cl_pt_apex.pt[1] > domain[1][0] and cl_pt_apex.pt[1] < domain[1][1]):
                    plt.text(cl_pt_apex.pt[0], cl_pt_apex.pt[1], str(bend.bend_evol_id), size=10)
            else:
                x,y = cl_pt_apex.pt[0], cl_pt_apex.pt[1]
                plt.text(x, y, str(bend.bend_evol_id), size=annot_text_size,
                         horizontalalignment="center")


def plot_bends0(ax, bends, domain = [[],[]], annotate = False,
               plot_apex = True, plot_inflex = False, plot_middle = False,
               plot_centroid = False, plot_centroid_trajec = False, plot_apex_trajec = True,
               plot_apex_proba = False, plot_normal = False, scale_normal = 1.,
               annot_text_size=10, color_bend = False, alpha=1,
               linewidth=1, markersize=2, cl_color=False):

    if cl_color:
        color = cl_color

    for i, bend in enumerate(bends):
        abscissa = []
        ordinates = []
        for cl_pt in bend.cl_points:
            abscissa += [cl_pt.pt[0]]
            ordinates += [cl_pt.pt[1]]

        color = 'r'
        if color_bend and bend.side == "up":
          color = 'b'

        ax.plot(abscissa, ordinates, linestyle='-', linewidth=linewidth, color=color, alpha=alpha)

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

        if plot_centroid and bend.isvalid:
            ax.plot(bend.pt_centroid[0], bend.pt_centroid[1],
                    marker = 'o', markeredgecolor='k', markerfacecolor='orange',
                    markersize = 0.8*markersize)
        if plot_apex_proba and bend.isvalid:
            x = []
            y = []
            for i, cl_pt in enumerate(bend.cl_points):
                x += [cl_pt.pt[0]]
                y += [cl_pt.pt[1]]
            ax.scatter(x, y,
                marker = 'o', c=bend.apex_probability, cmap='jet')

        if plot_normal:
            for cl_pt in bend.cl_points:
                plt.arrow(cl_pt.pt[0], cl_pt.pt[1],
                  cl_pt.data["Normal_x"]*scale_normal, cl_pt.data["Normal_y"]*scale_normal,
                  color='k', width=3, linewidth = 1)

        if annotate & ("cl_pt_apex" in bend.__dict__):
            if len(domain[0])>0:
                if (bend.cl_pt_apex.pt[0] > domain[0][0] and bend.cl_pt_apex.pt[0] < domain[0][1] and
                    bend.cl_pt_apex.pt[1] > domain[1][0] and bend.cl_pt_apex.pt[1] < domain[1][1]):
                    plt.text(bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1], str(bend.bend_evol_id), size=10)
            else:
                x,y = bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1]
                plt.text(x, y, str(bend.bend_evol_id), size=annot_text_size,
                         horizontalalignment="center")


def plot_apex_probability(bend_evol, plot_apex=False):
  plt.figure(dpi=150)
  for bend in bend_evol.bends:
    x = []
    y = []
    for i, cl_pt in enumerate(bend.cl_points):
        x += [cl_pt.pt[0]]
        y += [cl_pt.pt[1]]
    plt.scatter(x, y,
        marker = 'o', c=bend.apex_probability2, cmap='jet', s=4)
    plt.plot(x, y, 'k-', linewidth=0.5)
    if plot_apex:
        plt.plot(bend.cl_pt_apex.pt[0], bend.cl_pt_apex.pt[1], 'ko', markersize = 5, markerfacecolor='none')
  plt.axis('equal')
  plt.show()

def plot_section(section, ax, flow_dir=np.array([1,0]), norm_hor=1, norm_vert=1,
                 color_same_bend=True, colors=False, cmap=False):

    for i, isoline in enumerate(section.isolines):
        # coordinates to plot
        origin = section.isolines_origin[i]
        coords = cpf.points2coords(isoline.points)
        coords[0] = (origin[0] + coords[0]) / norm_hor
        coords[1] = (-origin[1] + coords[1]) / norm_vert

        if color_same_bend:
          color='b'
          style='--'
          if isoline.cl_pt_ref.data["Curvature"] < 0:
            style='--'
            color='r'
          ax.fill(coords[0], coords[1], linestyle=style, edgecolor=color,
                  fill=True, facecolor='w')
        else:
          color = 'b'
          if colors and cmap:
              color = cmap(colors(i))
          if isoline.cl_pt_ref.data["Curvature"] < 0:
            ax.fill(coords[0], coords[1], linestyle='--', edgecolor=color,
                    fill=True, facecolor='w')
          else:
            ax.fill(coords[0], coords[1], linestyle='--', edgecolor=color,
                    fill=True, facecolor='w')


def plot_versus_curvilinear(work_dir, abscissa, curves1, labels1, curves2 = [], labels2 = [], show = False):

    colors = ['b', 'g', 'm', 'c', 'lawngreen', 'purple', 'dodgerblue', 'r', 'orange', 'y', 'chocolate', 'gold', 'coral', 'hotpink']

    fig, ax1 = plt.subplots()

    for k, curve in enumerate(curves1):
        color = colors[k]
        ax1.plot(abscissa, curve, color, label = labels1[k])
        if k == 0:
            ax1.set_ylabel(labels1[0], color = 'k')
            for tl in ax1.get_yticklabels():
                tl.set_color('k')

    ax1.set_xlabel('Curvilinear abscissa (m)')
    if len(curves2) > 0:
        ax2 = ax1.twinx()
        for k, curve in enumerate(curves2):
            color = colors[7 + k]
            ax2.plot(abscissa, curve, color, label = labels2[k])
            if k == 0:
                ax2.set_ylabel(labels2[0], color='r')
                for tl in ax2.get_yticklabels():
                    tl.set_color('r')

    plt.ylim(-0.1,0.1)
    plt.tight_layout()
    if work_dir:
        pl.savefig(work_dir + 'props_versus_curv_abscissa.png', dpi = 300)

    if show:
        plt.show()

    pl.close('all')
    plt.close('all')

