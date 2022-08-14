# imports
import pandas as pd
import numpy as np

def meshToDataFrame(mesh):
    '''
    Given mesh, this function returns the meshes point arrays as a pandas dataframe.
    Data arrays that are vectors are placed into dataframe without editing the key.
    Data arrays with 3 components (which correspond to first derivatives) have their x and y components
    (column indices 0 and 1 respectively) placed into dataframe with the key "[array name]_x" or "[array name]_y".
    Data arrays with 9 components (with correspond to second derivatives) have their xx, xy, yx, and yy components
    (column indices 0, 1, 3, 4 respectively) placed into the dataframe with the key "[array name]_[component],"
    where component is xx, xy, etc.
    Additionally, a column of indices (0 indexed) is also added to the dataframe.
    '''
    ## create pandas dataframe
    mesh_data = pd.DataFrame()
    for key in mesh.point_data.keys():
        # vector
        if len(mesh.point_data[key].shape) == 1:
            mesh_data[key] = mesh.point_data[key]
        # 3 columns
        elif mesh.point_data[key].shape[1] == 3:
            mesh_data[key+"_x"] = mesh.point_data[key][:,0]
            mesh_data[key+"_y"] = mesh.point_data[key][:,1]
        # 9 columns
        elif mesh.point_data[key].shape[1] == 9:
            mesh_data[key+"_xx"] = mesh.point_data[key][:,0]
            mesh_data[key+"_xy"] = mesh.point_data[key][:,1]
            mesh_data[key+"_yx"] = mesh.point_data[key][:,3]
            mesh_data[key+"_yy"] = mesh.point_data[key][:,4]

    ## add indices to dataframe
    mesh_data["idx"] = np.arange(0, mesh_data.shape[0], 1)

    ## return dataframe
    return mesh_data
