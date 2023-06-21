from rotational_dataset import *
import numpy as np

# TODO make this a real test

num_points = 500
x, y, theta, h = get_base_data(num_points, cov=0.5)
displacement = get_displacements(x, y)

# A quick test to make sure the indexing has turned out as expected for the displacement.
test_disp = np.array([x[0]-x[1], y[0] - y[1]])
assert np.allclose(displacement[0, 1, :], test_disp)

alpha = compute_angular_components(theta, displacement)
out = compute_targets(h, alpha)

delta = np.pi/2
rx, ry, rtheta, rh = rotate_base_data(x, y, theta, h, delta)

# Do a quick test to check that the rotated unit vectors are as expected
u, ru = thetas_to_unit_vectors(theta), thetas_to_unit_vectors(rtheta)
test_u = np.zeros(u.shape[0])
for i in range(test_u.shape[0]):
    test_u[i] = u[i, :] @ ru[i, :]
assert np.allclose(np.arccos(test_u), delta)