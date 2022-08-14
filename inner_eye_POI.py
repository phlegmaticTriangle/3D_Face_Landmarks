# imports
import os
import pyvista as pv
import numpy as np
from mesh_to_DataFrame import meshToDataFrame
from calc_geo_quants import calcGeoQuant

def innerEyePOI(mesh):
    '''
    Given a 3D unnormalized mesh of a face, this function returns a 2-tuple.
    The first element of this tuple is an array of indices corresponding to the left inner eye POI.
    The second element of this tuple is an array of indices corresponding to the right inner eye POI.
    Both of these sets of indices should be used to index into mesh.points.
    '''
    ## numerical constants
    min_y = 7.5 # controls min dist above nosetip_y of POI
    max_y = 30 # controls max dist above nosetip_y of POI
    max_S = -0.375 # controls maximum S of POI
    filter_ratio = 0.1 # controls proportion of points that pass through filter for low first derivative magnitude
    prop_poi = 1/1000 # controls the number of POI returned in the end
    num_poi = int(mesh.points.shape[0] * prop_poi)

    ## get mesh's data as a pd dataframe
    mesh_data = meshToDataFrame(mesh)

    # find nosetip landmark point
    z_max_idx = np.argmax(mesh.points[:,2])
    nosetip = mesh.points[z_max_idx]

    nosetip_x = nosetip[0]
    nosetip_y = nosetip[1]

    ## inner eye must be sufficiently far from nose in x and y directions
    # y_range
    y = mesh_data.loc[:,'y']
    y_range = np.nanmax(y) - np.nanmin(y)

    # get points that are min_y% of y_range away from nosetip_y
    min_y_dist = min_y / 100 * y_range
    max_y_dist = max_y / 100 * y_range

    y_dist = np.abs(y - nosetip_y)

    y_far_enough = np.where(y_dist > min_y_dist, True, False)
    y_not_too_far = np.where(y_dist < max_y_dist, True, False)
    y_correct_dist = np.logical_and(y_far_enough, y_not_too_far)

    ## inner eye must be above the nose tip in y direction
    above_nosetip = np.where(y > nosetip_y, True, False)

    ## inner eye points must have a shape index in range [-1, -0.375]
    S_filter = np.where(mesh_data['S'] < max_S, True, False)

    ## get points that meet above requirments
    init_filter = np.logical_and(y_correct_dist, above_nosetip)
    init_filter = np.logical_and(init_filter, S_filter)

    mesh_data = mesh_data.iloc[init_filter]

    ## get points where z_x and z_y are approximately 0
    # i.e. get filter_ratio * mesh_data.shape[0] points with lowest z_square
    num_points = int(filter_ratio * mesh_data.shape[0])
    z_square = mesh_data.loc[:,'z_first_x'] ** 2 + mesh_data.loc[:,'z_first_y'] ** 2

    # partition and get points
    z_square_partition = np.argpartition(z_square, num_points)
    low_z_square_filter = z_square_partition[0:num_points]

    mesh_data = mesh_data.iloc[low_z_square_filter]

    ## partition the dataframe into a L and R dataframe before final selection
    L_mesh_data = mesh_data[mesh_data['x'] < nosetip_x]
    R_mesh_data = mesh_data[mesh_data['x'] > nosetip_x]

    ## select by K to get final points, selecting up to num_poi points in each dataframe

    # filter for L_mesh
    # only filter if enough points
    if (L_mesh_data.shape[0] > num_poi):
        percentile = (L_mesh_data.shape[0] - num_poi) / L_mesh_data.shape[0] * 100
        min_K = np.percentile(L_mesh_data['K'], percentile)
        L_mesh_data = L_mesh_data[L_mesh_data['K'] > min_K]
    
    # filter for R_mesh
    # only filter if enough points
    if (R_mesh_data.shape[0] > num_poi):
        percentile = (R_mesh_data.shape[0] - num_poi) / R_mesh_data.shape[0] * 100
        min_K = np.percentile(R_mesh_data['K'], percentile)
        R_mesh_data = R_mesh_data[R_mesh_data['K'] > min_K]

    ## return indices of L_mesh poi and R_mesh poi as np array
    return (L_mesh_data['idx'].to_numpy(), R_mesh_data['idx'].to_numpy())

def landmark_39(mesh):
    '''
    Return point indices that correspond to the inner eye corner poi for the L eye.
    '''
    return (innerEyePOI(mesh))[0]

def landmark_42(mesh):
    '''
    Return point indices that correspond to inner eye corner poi for the R eye.
    '''
    return (innerEyePOI(mesh))[1]

if __name__ == "__main__":
    # test 
    mesh_path = os.path.joing("face_scans", "Serag_Rest.stl")
    mesh = pv.read(mesh_path)
    unnormal_mesh, normal_mesh = calcGeoQuant(mesh)
    left_poi, right_poi = innerEyePOI(unnormal_mesh)
    poi_indices = np.append(left_poi, right_poi)
    p = pv.Plotter()
    p.add_mesh(mesh, color = 'w')
    p.add_mesh(pv.PointSet(mesh.points[poi_indices]), color = 'r', render_points_as_spheres= True, point_size= 10)
    p.show()