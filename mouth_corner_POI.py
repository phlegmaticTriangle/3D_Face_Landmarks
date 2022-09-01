# imports
import numpy as np
import os
import pyvista as pv
from mesh_to_DataFrame import meshToDataFrame
from calc_geo_quants import calcGeoQuant

def outerMouthPOI(mesh):
    '''
    Given a mesh with data arrays that are not normalized, this function returns a 2-tuple of arrays of indices.
    The first tuple is the array for the POI for the left outer mouth corner.
    The second tuple is the array for POI for the right outer mouth corner.
    These arrays should be used to index into mesh.points to get POI points.
    '''
    # constants
    S_max = 0 # max shape index
    S_min = -1 # min shape index
    C_prop = 0.5 # proportion of points by Curvedness to return
    min_y_diff = 10
    min_x_diff = 10
    z_prop = 75

    # create dataframe from mesh
    mesh_df = meshToDataFrame(mesh)

    # find nosetip
    nosetip_idx = np.argmax(mesh_df.loc[:, 'z'])
    nosetip_x = mesh.points[nosetip_idx][0]
    nosetip_y = mesh.points[nosetip_idx][1]

    # only consider points sufficiently far in the x direction
    x_range = np.nanmax(mesh_df.loc[:, 'x']) - np.nanmin(mesh_df.loc[:, 'x'])
    x_dist = np.abs(mesh_df.loc[:, 'x'] - nosetip_x)
    far_enough_x = np.where(x_dist > min_x_diff * x_range / 100, True, False)
    mesh_df = mesh_df.loc[far_enough_x, :]

    # only consider points sufficiently below the nosetip
    y_range = np.nanmax(mesh_df.loc[:, 'y']) - np.nanmin(mesh_df.loc[:, 'y'])
    below_nosetip = np.where(mesh_df.loc[:, 'y'] < nosetip_y - min_y_diff / 100 * y_range, True, False)
    mesh_df = mesh_df.loc[below_nosetip, :]

    # only consider points within minimum_z + z_range * z_prop / 100
    min_z = np.nanmin(mesh_df.loc[:, 'z'])
    max_z = np.nanmax(mesh_df.loc[:, 'z'])
    min_accept_z = min_z + (max_z - min_z) * z_prop / 100
    large_enough_z = np.where(mesh_df.loc[:, 'z'] > min_accept_z, True, False)
    mesh_df = mesh_df.loc[large_enough_z, :]
   
    # filter df by shape index
    S_less_max = np.where(mesh_df.loc[:, 'S'] < S_max, True, False)
    S_greater_min = np.where(mesh_df.loc[:, 'S'] > S_min, True, False)

    correct_S = np.logical_and(S_less_max, S_greater_min)

    mesh_df = mesh_df.loc[correct_S, :]

    # split into left and right of nosetip

    left_nosetip = np.where(mesh_df.loc[:, 'x'] < nosetip_x, True, False)
    right_nosetip = np.where(mesh_df.loc[:, 'x'] > nosetip_x, True, False)

    left_df = mesh_df.loc[left_nosetip, :]
    right_df = mesh_df.loc[right_nosetip, :]

    # get top num_poi points with respect to C for each or all points if number of points is less than num_poi
    left_df = left_df.sort_values('C', axis = 0, ascending=False)

    right_df = right_df.sort_values('C', axis = 0, ascending=False)

    left_df = left_df.iloc[0:int(C_prop * left_df.shape[0]), :]
    right_df = right_df.iloc[0:int(C_prop * right_df.shape[0]), :]

    left_indices = left_df.loc[:, 'idx']
    right_indices = right_df.loc[:, 'idx']

    return left_indices, right_indices

def landmark_48(mesh):
    '''
    Returns the point indices that correspond with left mouth corner POI.
    '''
    return (outerMouthPOI(mesh))[0]

def landmark_54(mesh):
    '''
    Returns the point indices that correspond with right mouth corner POI.
    '''
    return (outerMouthPOI(mesh))[1]


if __name__ == "__main__":
    # test
    mesh_path = os.path.join("face_scans", "demo.stl")
    mesh = pv.read(mesh_path)
    unnormal_mesh, normal_mesh = calcGeoQuant(mesh)
    left_poi, right_poi = outerMouthPOI(unnormal_mesh)
    p = pv.Plotter()
    p.add_mesh(unnormal_mesh, color = 'w')
    p.add_mesh(pv.PointSet(mesh.points[left_poi]), color = 'r', render_points_as_spheres= True, point_size= 10)
    p.add_mesh(pv.PointSet(mesh.points[right_poi]), color = 'r', render_points_as_spheres= True, point_size= 10)
    p.show()