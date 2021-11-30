function [mesh, rendering_parameters] = fit_shape_and_pose(morphable_model, ...
    blendshapes, landmarks, landmark_mapper, image_width, image_height, ...
    edge_topology, contour_landmarks, model_contour, num_iterations, ...
    num_shape_coefficients_to_fit, lambda)
% FIT_SHAPE_AND_POSE  Fit a 3DMM shape model to landmarks.
%   [ mesh, rendering_parameters ] = FIT_SHAPE_AND_POSE(morphable_model, ...
%     blendshapes, landmarks, landmark_mapper, image_width, image_height, ...
%     edge_topology, contour_landmarks, model_contour, num_iterations, ...
%     num_shape_coefficients_to_fit, lambda)
%
%   This function fits a 3D Morphable Model to landmarks in an image.
%   It fits the pose (camera), PCA shape model, and expression blendshapes
%   in an iterative way.
%
%   landmarks must be a 68 x 2 matrix with ibug landmarks, in the order
%   from 1 to 68.
%
%   Default values for some of the parameters: num_iterations = 5,
%   num_shape_coefficients_to_fit = all (-1), and lambda = 30.0.
%
%   Please see the C++ documentation for the description of the parameters:
%   http://patrikhuber.github.io/eos/doc/ (TODO: Update to v0.9.1!)
%
%   NOTE: In contrast to the C++ function, this Matlab function expects the
%   morphable_model, blendshapes, landmark_mapper, edge_topology,
%   contour_landmarks and model_contour as *filenames* to the respective
%   files in the eos/share/ directory, and not the objects directly.

if(~isa(landmarks,'double'))
    error('Please specify the landmarks as type double.');
end

% We'll use default values to the following arguments, if they're not
% provided:
if (~exist('edge_topology', 'var')), edge_topology = '../share/sfm_3448_edge_topology.json'; end
if (~exist('contour_landmarks', 'var')), contour_landmarks = '../share/ibug_to_sfm.txt'; end
if (~exist('model_contour', 'var')), model_contour = '../share/sfm_model_contours.json'; end
if (~exist('num_iterations', 'var')), num_iterations = 5; end
if (~exist('num_shape_coefficients_to_fit', 'var')), num_shape_coefficients_to_fit = -1; end
if (~exist('lambda', 'var')), lambda = 30.0; end

[ mesh, rendering_parameters ] = fitting(morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda);

end
