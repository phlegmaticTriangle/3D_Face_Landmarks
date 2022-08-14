# imports
import pyvista as pv
import numpy as np
import scipy as sp
import scipy.stats 
from scipy import spatial

def calcGeoQuant(mesh):
    '''
    Given a face mesh, this function calculates the relevant geometric quantities of the mesh and returns this data in two forms.
    The first form is a .vtk with data arrays that are not normalized, and the second form is a .vtk with normalized data arrays.
    '''
    # force mesh points to be stored as float64
    mesh.points = mesh.points.astype('float64')

    # store z as data array
    mesh.point_data['z'] = mesh.points[:,2]

    # project mesh to x-y planes
    project_mesh = mesh.copy(deep = True)
    project_mesh.point_data['z'] = project_mesh.points[:,2]
    project_mesh.points[:,2] = np.zeros_like(project_mesh.points[:,2])

    ## calculate first derivative of z
    project_mesh = project_mesh.compute_derivative(scalars='z',\
        gradient='z_first')
    
    ## calculate second derivative of z
    project_mesh = project_mesh.compute_derivative(scalars='z_first',\
        gradient='z_second')
    
    ## calculate first fundamental form 
    project_mesh.point_data['E'] = 1 + project_mesh.point_data['z_first'][:,0] ** 2
    project_mesh.point_data['F'] = project_mesh.point_data['z_first'][:,0]*\
        project_mesh.point_data['z_first'][:,1]
    project_mesh.point_data['G'] = 1 + project_mesh.point_data['z_first'][:,1] ** 2

    ## calculate second fundamental form 
    first_deriv = project_mesh.point_data['z_first']
    second_deriv = project_mesh.point_data['z_second']
    second_fund_form_denom = np.sqrt(first_deriv[:,0] ** 2 + first_deriv[:,1]** 2 + 1)
    project_mesh.point_data['e'] = second_deriv[:,0] / second_fund_form_denom
    project_mesh.point_data['f'] = second_deriv[:,1] / second_fund_form_denom
    project_mesh.point_data['g'] = second_deriv[:,4] / second_fund_form_denom

    ## calculate K and H
    e_first = project_mesh.point_data['E']
    f_first = project_mesh.point_data['F']
    g_first = project_mesh.point_data['G']
    e = project_mesh.point_data['e']
    f = project_mesh.point_data['f']
    g = project_mesh.point_data['g']
    project_mesh.point_data['K'] = (e*g - f**2) / (e_first * g_first - f_first **2)
    project_mesh.point_data['H'] = (e*g_first - 2 * f*f_first + e_first*g) / (2*(e_first * g_first - f_first **2))

    ## calculate k_min and k_max
    project_mesh.point_data['k_max'] = project_mesh.point_data['H'] +\
                np.sqrt(np.clip(project_mesh.point_data['H'] ** 2 - project_mesh.point_data['K'], a_min = 0, a_max = None))
    project_mesh.point_data['k_min'] = project_mesh.point_data['H'] -\
                np.sqrt(np.clip(project_mesh.point_data['H'] ** 2 - project_mesh.point_data['K'], a_min = 0, a_max = None))

    ## calculate first derivative of fundamental form coefficients
    # E deriv
    project_mesh = project_mesh.compute_derivative(scalars='E', gradient='E_first')
    # E deriv
    project_mesh = project_mesh.compute_derivative(scalars='F', gradient='F_first')
    # G deriv
    project_mesh = project_mesh.compute_derivative(scalars='G', gradient='G_first')
    # e deriv
    project_mesh = project_mesh.compute_derivative(scalars='e', gradient='e_first')
    # f deriv
    project_mesh = project_mesh.compute_derivative(scalars='f', gradient='f_first')
    # g deriv
    project_mesh = project_mesh.compute_derivative(scalars='g', gradient='g_first')

    ## calculate second derivative of fundamental form coefficients
    # second derivative of E
    project_mesh = project_mesh.compute_derivative(scalars='E_first', gradient='E_second')

    # second derivative of F
    project_mesh = project_mesh.compute_derivative(scalars='F_first', gradient='F_second')

    # second derivative of G
    project_mesh = project_mesh.compute_derivative(scalars='G_first', gradient='G_second')

    # second derivative of e
    project_mesh = project_mesh.compute_derivative(scalars='e_first', gradient='e_second')

    # second derivative of f
    project_mesh = project_mesh.compute_derivative(scalars='f_first', gradient='f_second')

    # second derivative of g
    project_mesh = project_mesh.compute_derivative(scalars='g_first', gradient='g_second')

    ## calculating shape index and curvedness index
    curvature_sum = project_mesh.point_data['k_max'] + project_mesh.point_data['k_min']
    curvature_diff = project_mesh.point_data['k_max'] - project_mesh.point_data['k_min']
    project_mesh.point_data['S'] = -2 / np.pi * np.arctan2(curvature_sum, curvature_diff) # use arctan2 for division

    # curvedness index
    project_mesh.point_data['C'] = np.sqrt(1/2 * (project_mesh.point_data['k_max'] ** 2 + project_mesh.point_data['k_min']**2))      

    ## add x and y information to point data
    project_mesh.point_data['x'] = mesh.points[:,0]
    project_mesh.point_data['y'] = mesh.points[:,1]

    ## return project_mesh into 3D version of itself
    project_mesh.points[:,2] = project_mesh['z']

    ## set unnormalized mesh
    unnormalized_mesh = project_mesh.copy(deep=True)

    ## normalize mesh 
    ## iterate through all data arrays associated with the points
    # get point array names
    arr_names = project_mesh.point_data.keys()
    # z-score calculation for each array
    for arr in arr_names:
        # calculate z-score along axis 0
        arr_z_score = scipy.stats.zscore(project_mesh.point_data[arr], axis = 0)

        # replace arr with its z_score array
        project_mesh.point_data[arr] = arr_z_score

    normalized_mesh = project_mesh

    return unnormalized_mesh, normalized_mesh