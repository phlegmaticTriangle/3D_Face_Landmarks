# imports
import os
import joblib
import pyvista as pv
import argparse
import numpy as np
from rotation_preprocessing import orient_mesh
from calc_geo_quants import calcGeoQuant
from inner_eye_POI import landmark_39
from inner_eye_POI import landmark_42
from mouth_corner_POI import landmark_48
from mouth_corner_POI import landmark_54
from landmark_refinement import refineLandmark


def main(mesh_path = os.path.join("face_scans", "demo.stl"), reorient = False):
    '''
    Given the path of a face scan, this function will find five landmarks on the face scan: the pronasal, the endocanthions, 
    and the cheilions. If reorient is True, the function will try to orient the face mesh into the canonical position first.
    The output of this function is a .txt file with the locations of the landmarks. 
    The order of the landmarks in the file is as follows: pronasal, left endocanthion, right endocanthion, left cheilion,
    and right cheilion. The function also displays a visualization of the locations of the landmarks on the face scan.
    '''
    ## get face mesh
    mesh = pv.read(mesh_path)

    ## reorient if specified
    if (reorient):
        mesh = orient_mesh(mesh)

    ## calculate geometric quantities of mesh
    mesh, normal_mesh = calcGeoQuant(mesh)

    ## create output file and list of landmarks
    if not os.path.exists("results"):
        os.mkdir("results")
    output_file = open(os.path.join("results", "landmarks.txt"), "w")
    landmark_list = []

    ## find tip of the nose
    nosetip_idx = np.argmax(mesh.points[:, 2])
    nosetip = mesh.points[nosetip_idx]
    # write nosetip to output_file and save in landmark_list
    output_file.write(str(nosetip[0]) + "\t" + str(nosetip[1]) + "\t" + str(nosetip[2]) + "\n")
    landmark_list.append(nosetip)

    ## find eye and mouth corners
    # list of landmark numbers
    landmark_ids = [39, 42, 48, 54]
    landmark_functions = [landmark_39, landmark_42, landmark_48, landmark_54]
    # iterate through landmarks
    for idx, landmark_id in enumerate(landmark_ids):
        # get POI
        poi_indices = landmark_functions[idx](mesh)
        # get model corresponding with current landmark
        model = joblib.load(os.path.join("models", "landmark_" + str(landmark_id) + ".joblib"))
        # created excluded_cols list
        excluded_cols = ['idx']
        if landmark_id == 48 or landmark_id == 54:
            excluded_cols.append('y')
        # refine landmark into final predicted location
        final_loc = refineLandmark(normal_mesh, poi_indices, model, excluded_cols)
        # write final_loc to output_file and save in landmark_list
        output_file.write(str(final_loc[0]) + "\t" + str(final_loc[1]) + "\t" + str(final_loc[2]) + "\n")
        landmark_list.append(final_loc)
    
    ## plot the five found landmarks on the face scan
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color = 'w')
    plotter.add_mesh(pv.PointSet(np.asarray(landmark_list)), color = 'r', render_points_as_spheres = True, point_size = 20)
    plotter.show()

    ## close file
    output_file.close()


if __name__ == "__main__":
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reorient", help="calls program to orient face into standard orientation", action="store_true", default=False)
    parser.add_argument("--mesh", help="specifies the path to the face mesh", default=os.path.join("face_scans", "demo.stl"))
    args = parser.parse_args()
    ## call main
    main(args.mesh, args.reorient)