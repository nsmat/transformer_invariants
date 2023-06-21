import numpy as np
import pandas as pd


def get_base_data(num_points, cov=0.5):
    mu = np.zeros(2)
    sigma = np.array([[1, cov], [cov, 1]])
    locations = np.random.multivariate_normal(
        mu, sigma, size=(num_points)
    )
    x, y = locations[:, 0], locations[:, 1]
    angles = np.random.uniform(0, np.pi * 2, num_points)
    hidden_features = np.random.normal(0, 1, size=num_points)

    return x, y, angles, hidden_features


def get_displacements(xs, ys):
    x_displacement = np.subtract.outer(xs, xs)
    y_displacement = np.subtract.outer(ys, ys)

    displacement_vectors = np.dstack((x_displacement, y_displacement))

    return displacement_vectors


def thetas_to_unit_vectors(thetas):
    x, y = np.cos(thetas), np.sin(thetas)
    return np.column_stack((x, y))


def compute_angular_components(thetas, displacement_vectors):
    # Define \alpha_ij = (x_i - x_j) @ theta_j
    # Where theta_j is the unit vector pointing in the direction theta
    theta_j = thetas_to_unit_vectors(thetas)

    num_points = thetas.shape[0]
    alphas = np.zeros(shape=(num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            alphas[i, j] = displacement_vectors[i, j] @ theta_j[j]

    return alphas


def compute_targets(hidden, angular_components):
    return angular_components @ hidden


def compute_targets_from_base_data(x, y, angles, hidden_features):
    displacement = get_displacements(x, y)
    alpha = compute_angular_components(angles, displacement)
    output_forces = compute_targets(hidden_features, angular_components=alpha)

    return output_forces


def rotate_base_data(x, y, angles, hidden_features, r_theta):
    loc_matrix = np.row_stack((x, y))
    c, s = np.cos(r_theta), np.sin(r_theta)
    rotation_matrix = np.array([[c, s], [-s, c]])  # anticlockwise rotation
    out_loc = rotation_matrix @ loc_matrix
    out_x, out_y = out_loc[0, :], out_loc[1, :]

    out_angles = angles + r_theta % (np.pi * 2)
    return out_x, out_y, out_angles, hidden_features


def translate_base_data(x, y, angles, hidden_features, translation_x, translation_y):
    tx, ty = x + translation_x, y + translation_y
    return tx, ty, angles, hidden_features


def make_one_graph(num_points=120, rot=None, translation=None):
    assert not (rot and translation)  # TODO hack - need to make these affine transformations somehow?
    x, y, theta, h = get_base_data(num_points, cov=0.5)

    if rot:
        x, y, theta, h = rotate_base_data(x, y, theta, h, rot)
    if translation:
        x, y, theta, h = translate_base_data(x, y, theta, h, rot)

    output_forces = compute_targets_from_base_data(x, y, theta, h)

    data = {'x': x, 'y': y, 'theta': theta, 'h': h, 'output_forces': output_forces}
    df = pd.DataFrame(data=data)
    df = df.reset_index().rename(columns={'index': 'vertex_id'})

    return df

def make_dataset(num_graphs, num_points=500):
    frames = []
    for n in range(num_graphs):
        data = make_one_graph(num_points)

        df = pd.DataFrame(data)
        df['graph_id'] = n
        frames.append(df)
    return pd.concat(frames)
