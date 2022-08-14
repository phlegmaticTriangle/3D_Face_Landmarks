import numpy as np
from sklearn.cluster import OPTICS
from scipy.spatial import KDTree
from mesh_to_DataFrame import meshToDataFrame

def refineLandmark(normal_mesh, POI_indices, model, excluded_cols = ['idx']):
    '''
    Given a mesh with normalized data arrays, a set of indices corresponding with POI for a given landmark, 
    a model that regresses each POI's distance from the given landmark, 
    and a list of columns and/or data arrays relating to normal_mesh that are not accepted by the model
    this function returns the coordinates of the predicted location of the landmark as a numpy array with shape (3, ).
    '''
    ## constants
    prop = 1/2 # proportion of points all points needed to form a cluster in OPTICS

    ## create DataFrame from the normal_mesh's data arrays
    normal_df = meshToDataFrame(normal_mesh)

    ## get DataFrame corresponding to POI points
    POI_df = normal_df.iloc[POI_indices]
    
    ## drop excluded_cols
    for col in excluded_cols:
        POI_df = POI_df.drop(columns = col)

    ## regress the distance of each POI from the actual landmark location
    poi_dist = model.predict(POI_df.to_numpy())

    ## cluster POI based on their location in face scan
    POI_points = normal_mesh.points[POI_indices]
    clust = OPTICS(min_samples = prop)
    clust = clust.fit(POI_points)
    # move clustered points into a dictionary based on their cluster
    POI_clusts = {}
    for poi_idx, clust_idx in enumerate(clust.labels_):
        if clust_idx != -1: # point not in a cluster
            if clust_idx in POI_clusts.keys(): # if an entry for the cluster of the point at poi_idx already exists
                POI_clusts[clust_idx].append(poi_idx)
            else: # create a new dictionary entry for this point's cluster
                POI_clusts[clust_idx] = [poi_idx]
    
    ## get median x, y, z coordinates of the cluster with the minimum median predicted distance 
    min_pred_dist = float('inf')
    min_clust_med = np.zeros(3)
    for key in POI_clusts.keys():
        # get median distance of clust with key key
        clust_indices = POI_clusts[key]
        med_dist = np.median(poi_dist[clust_indices])
        # get median x, y, z coordinates
        clust_med = np.median(POI_points[clust_indices], axis = 0)
        # update min if needed
        if med_dist < min_pred_dist:
            min_pred_dist = med_dist
            min_clust_med = clust_med
    
    ## project median x, y, z coordinates of the cluster with the minimum median predicted distance onto the face mesh
    kd_tree = KDTree(normal_mesh.points)
    landmark_idx = (kd_tree.query(min_clust_med))[1]
    pred_landmark = normal_mesh.points[landmark_idx]

    ## return predicted landmark location
    return pred_landmark







