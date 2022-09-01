# imports
import pyvista as pv
import os
import numpy as np
from scipy import optimize
import scipy as sp

# find z-direction functions
def z_direc(mesh):
    '''
    Given a face mesh, this function finds the orientation of the vector that comes straight out of the face.
    '''
    # create initial guess
    init_vect = np.random.rand(3)
    init_vect = init_vect / np.linalg.norm(init_vect)
    # find z_direc
    res = optimize.dual_annealing(z_cost, bounds=[[-1,1],[-1,1],[-1,1]], args=(mesh,),\
            maxiter=150, x0 = init_vect, accept=-100)
    return (res.x) / np.linalg.norm(res.x) # return result

def z_cost(z_prime_vect, face_mesh): 
    '''
    Finds the negative average sum of the new z components of normal vectors after the change of base
    '''
    # normalize z_prime_vect
    z_prime_vect = z_prime_vect / np.linalg.norm(z_prime_vect)
    return -1 * np.sum(face_mesh.point_normals @ z_prime_vect) / (face_mesh.point_normals.shape[0]) 

# find x-direction functions
def x_direc(mesh):
    '''
    Given a face mesh, this function finds the orientation of the vector that is horizontal within the plane of the face.
    '''
    # initialize guess
    init_vect = np.random.rand(3)
    init_vect = init_vect / np.linalg.norm(init_vect)
    # find x_direc
    res = optimize.dual_annealing(x_cost, bounds=[[-1,1],[-1,1],[-1,1]], args=(mesh, ),\
            x0 = init_vect, maxiter = 150, accept=-100)

    return (res.x) / np.linalg.norm(res.x)

def x_cost(x_prime_vect, mesh):
    '''
    Given a proposed x_direc and a face mesh, determines the cost of the x_prime_vect 
    by determining how similar the distribution of normal vector components in this direction is
    to an ideal distribution, which is a secant function in this case.
    '''
    # constants
    num_bins = 20 # number of bins to use in the histogram when calculating cost
    secant_const = 1/7 # scaling factor for the secant modeling the ideal distribution

    unit_norms = mesh.point_normals / np.transpose(np.tile(np.linalg.norm(mesh.point_normals, axis=1), (3, 1))) # unit normals
    x_prime_vect = x_prime_vect / np.linalg.norm(x_prime_vect) # normalize x'
    product_mat = unit_norms @ x_prime_vect

    # create histogram
    hist, bin_edges = np.histogram(product_mat, num_bins, density=True) # density function s.t. integral = 1

    # finding cost
    # creating vector of points in between bin edges (where diff to ideal bowl shape will be evaluated)
    bin_diffs = np.diff(bin_edges)
    bin_diffs /= 2 # find half of the diff to next bin
    bin_points = bin_edges[0:(bin_edges.shape[0] - 1)] + bin_diffs # cut off last element of bin_edges, add bin_diffs

    # comparing to ideal secant shape and returning as cost
    ideal_vals = secant_const * (1 / np.cos(bin_points * np.pi / 2))
    real_ideal_diff_non_squared = hist - ideal_vals
    return np.sum(real_ideal_diff_non_squared ** 2)

# functions for orienting the mesh into a standard position

def get_rot_matrix(mesh):
    '''
    Given a mesh, this function finds the x, y, and z directions and returns the rotation matrix
    necessary to bring the mesh back into the standard position.
    '''
    ## decimate the mesh for faster processing
    target_cells= 5000
    num_triangles = mesh.n_faces
    reduction_factor = 1 - target_cells / num_triangles
    decimated_mesh = mesh.decimate_pro(reduction_factor)
    decimated_mesh = decimated_mesh.compute_normals()
  
    
    ## ensure that normal vectors are pointed away from center of mass
    mesh_COM = decimated_mesh.center_of_mass()
    to_COM_vects = mesh_COM - decimated_mesh.points
    dot_products = np.diagonal(decimated_mesh.point_normals @ np.transpose(to_COM_vects))
    flip = np.where(dot_products > 0, True, False)
    keep = np.logical_not(flip)
    new_normals = np.zeros_like(decimated_mesh.point_data['Normals'])
    new_normals[flip] = -1 * decimated_mesh.point_data['Normals'][flip]
    new_normals[keep] = decimated_mesh.point_data['Normals'][keep]
    decimated_mesh.point_data['Normals'] = new_normals

    ## find x_direction
    x_direction = x_direc(decimated_mesh)

    ## find z_direction
    z_direction = z_direc(decimated_mesh)
    
    ## force z to be orthoganal with x
    z_direction -= np.dot(x_direction, z_direction) * x_direction
    z_direction = z_direction / np.linalg.norm(z_direction)

    ## choose y_direction as x_direction cross z_direction (another valid choice would be negative cross product)
    y_direction = np.cross(x_direction, z_direction)
    y_direction = y_direction / np.linalg.norm(y_direction)

    ## transform points based on 3 directions into an initial guess correct final position
    # create a rotation matrix
    rotate_mat = np.zeros((3,3))
    rotate_mat[:, 0] = x_direction
    rotate_mat[:, 1] = y_direction
    rotate_mat[:, 2] = z_direction
    sp.linalg.inv(rotate_mat, overwrite_a=True)

    return rotate_mat

def orient_mesh(mesh):
    '''
    Given a complete mesh in pyvista polydata format in an arbitrary orientation, 
    this function will return the mesh after it has been positioned into a standard/canonical position.
    This standard/canonical position is described as follows:
    The x-direction is horizontal in the face; the y-direction is vertical in the face;
    the z-direction comes directly out of the face and directly out of the x-y plane.
    '''
    ## constants
    tol = 0.05 # the maximum difference from the identity matrix for convergence of rotation matrix to be satisfied
    prop = 0.2 # the proportion of the range of x values and range of y values
                # that determines the minimum distance of the nosetip from the edges of the mesh
    y_prop = 0.04 # determines the range of y values around the nosetip to consider when calculating z_first_y
    x_prop = 0.04 # determines the range of x values around the nosetip to consider when calculating z_first_y
    large_float = 1e30 # used as a substitute for infinitiy when calulating z_first_y

    ## rotate the mesh until its position has converged
    converged = False
    ideal_rotate_mat = np.eye(3)
    while not converged:
        ## get rotation matrix
        rotate_mat = get_rot_matrix(mesh)

        ## rotate the points using the rotations matrix
        mesh_points = np.transpose(mesh.points)
        mesh_points = rotate_mat @ mesh_points
        mesh.points = np.transpose(mesh_points)

        ## check for convergence
        rotate_mat_diff = np.abs(rotate_mat) - ideal_rotate_mat
        if np.linalg.norm(rotate_mat_diff) < tol:
            converged = True
        
    ## initial attempt to find the nosetip as the point with the highest z-value
    nosetip_idx = np.argmax(mesh.points[:, 2])
    
    ## check if the nosetip has been found correctly
    ## in particular, check to make sure normal vectors were oriented properly
    ## if they weren't, then nosetip should be near edge of the face

    # check if nosetip within 20% of x_range of x_min or x_max
    x_nosetip = mesh.points[nosetip_idx][0]
    x_range = np.nanmax(mesh.points[:,0]) - np.nanmin(mesh.points[:,0])
    close_to_x_border = (abs(x_nosetip - np.nanmax(mesh.points[:,0])) < prop*x_range) or \
        (abs(x_nosetip - np.nanmin(mesh.points[:,0])) < prop*x_range)

    # check if nosetip within 20% of y_range of y_min or y_max
    y_nosetip = mesh.points[nosetip_idx][1]
    y_range = np.nanmax(mesh.points[:,1]) - np.nanmin(mesh.points[:,1])
    close_to_y_border = (abs(y_nosetip - np.nanmax(mesh.points[:,1])) < prop*y_range) or \
        (abs(y_nosetip - np.nanmin(mesh.points[:,1])) < prop*y_range)

    if (close_to_y_border or close_to_x_border):
        flip_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # flip the points using the flip matrix
        mesh_points = np.transpose(mesh.points)
        mesh_points = flip_mat @ mesh_points
        mesh.points = np.transpose(mesh_points)

        # find nosetip once more
        nosetip_idx = np.argmax(mesh.points[:,2])

    ## find the correct y-orientation using derivatives

    # get all points within these bounds
    nosetip_idx = np.argmax(mesh.points[:, 2])
    x_nosetip = mesh.points[nosetip_idx][0]
    y_nosetip = mesh.points[nosetip_idx][1]
    y_range = np.nanmax(mesh.points[:, 1]) - np.nanmin(mesh.points[:, 1])
    x_range = np.nanmax(mesh.points[:, 0]) - np.nanmin(mesh.points[:, 0])

    within_y = np.where(np.abs(mesh.points[:, 1] - y_nosetip) < y_prop*y_range, True, False)
    within_x = np.where(np.abs(mesh.points[:, 0] - x_nosetip) < x_prop*x_range, True, False)
    within_bounds = np.logical_and(within_x, within_y)

    above_nosetip = np.where(mesh.points[:, 1] > y_nosetip, True, False)
    below_nosetip = np.where(mesh.points[:, 1] < y_nosetip, True, False)

    within_and_above = np.logical_and(within_bounds, above_nosetip)
    within_and_below = np.logical_and(within_bounds, below_nosetip)

    indices_vect = np.arange(0, mesh.points.shape[0])
    within_and_above_idx = indices_vect[within_and_above]
    within_and_below_idx = indices_vect[within_and_below]

    # calculate z_first_y for the points within bounds using normal vect
    # calculating for points above the nosetip
    above_dividend = mesh.point_normals[within_and_above_idx][:, 1]
    above_divisor = mesh.point_normals[within_and_above_idx][:, 2]
    above_z_first_y = -1 * np.divide(above_dividend, above_divisor, out=np.full(above_dividend.shape, large_float).astype('float64'), \
        where = (above_divisor!=0))
    above_avg_z_first_mag = np.mean(np.abs(above_z_first_y))

    # calculating for points below the nosetip
    below_dividend = mesh.point_normals[within_and_below_idx][:, 1]
    below_divisor = mesh.point_normals[within_and_below_idx][:, 2]
    below_z_first_y = -1 * np.divide(below_dividend, below_divisor, out=np.full(below_dividend.shape, large_float).astype('float64'), \
        where = (below_divisor!=0))
    below_avg_z_first_mag = np.mean(np.abs(below_z_first_y))

    # compare the magnitudes of z_first_y medians
    correct_y = below_avg_z_first_mag > above_avg_z_first_mag

    # if not correct, reverse y_direction
    if not correct_y:
        # get reflection matrix
        x_direction = np.array([1, 0, 0])
        y_direction = np.array([0, -1, 0])
        z_direction = np.array([0, 0, 1])
        reflect_mat = np.zeros((3,3))
        reflect_mat[:, 0] = x_direction
        reflect_mat[:, 1] = y_direction
        reflect_mat[:, 2] = z_direction

        # reflect the points using the reflection matrix
        mesh_points = np.transpose(mesh.points)
        mesh_points = reflect_mat @ mesh_points
        mesh.points = np.transpose(mesh_points)
    
    ## clear data before return
    mesh.clear_data()

    ## return rotated mesh
    return mesh

if __name__ == "__main__":
    ## test orient mesh

    # get mesh
    mesh_path = os.path.join("face_scans", "demo.stl")
    mesh = pv.read(mesh_path)

    # randomize its orientations
    
    random_orient = sp.linalg.orth(np.random.rand(3, 3))
    mesh_points = np.transpose(mesh.points)
    mesh_points = random_orient @ mesh_points 
    mesh.points = np.transpose(mesh_points) 
    
    print(random_orient)
    
    # show the randomly oriented mesh
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.add_axes()
    p.show()

    # orient the mesh
    mesh = orient_mesh(mesh)

    # show the oriented mesh
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.add_axes()
    p.show()