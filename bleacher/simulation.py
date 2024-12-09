import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Function to generate points based on the provided width, height, dx, and dy
def generate_points(width, height, dx, dy):
    # Calculate the number of points along x and y based on dx and dy
    num_x = max(1, int(np.floor(width / dx)))  # Ensure at least 1 point in x direction
    num_y = max(1, int(np.floor(height / dy)))  # Ensure at least 1 point in y direction

    # Generate the coordinates, centered at (0, 0)
    if num_x == 1:
        x_coords = np.array([0])  # Single column in the middle
    else:
        x_coords = np.linspace(- (num_x // 2) * dx, (num_x // 2) * dx, num_x)

    if num_y == 1:
        y_coords = np.array([0])  # Single row in the middle
    else:
        y_coords = np.linspace(- (num_y // 2) * dy, (num_y // 2) * dy, num_y)

    # Create the 2D grid of points
    points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    return points

# Function to make the points using generate_point with depth
def generate_points_depth(width, height, depth, dx, dy):
    points = generate_points(width, height, dx, dy)
    
    # Convert to the points with z coordinate
    points_3d = np.column_stack((
        points[:, 0],  # x coordinate
        points[:, 1],  # y coordinate
        np.full(points.shape[0], depth)
    ))

    return points_3d

# Function to update the plot based on user input
def plot_grid(width, height, dx, dy):
    points = generate_points(width, height, dx, dy)
    
    # Clear the previous plot
    plt.figure(figsize=(6, 6))
    
    # Plot the points
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    
    # Set the limits of the plot based on the board dimensions
    plt.xlim(-width / 2, width / 2)
    plt.ylim(-height / 2, height / 2)
    
    # Set labels and title
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(f"Grid with dx={dx}, dy={dy}, width={width}, height={height}")
    
    # Show the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

# Function to calculate the peak candela from lumen
def calculate_i0_from_phi(phi, illumination='lambertian'):
    # check if the illumination input is correct.
    if illumination!='lambertian' and illumination!='point':
        raise ValueError("The illumination should be either lambertian or point.")

    if illumination=='point':
        i0 = phi / (4*np.pi)
    else:
        i0 = phi / (np.pi)
    
    return i0
    
# Function to compute the inverse square distance from observation point to each grid point
def compute_sum_at_obs_point(points, obs_point, phi, unit, illumination='lambertian'):
    # check if the illumination input is correct.
    if illumination!='lambertian' and illumination!='point':
        raise ValueError("The illumination should be either lambertian or point.")
        
    # Compute the vector from each point to the observation point
    vectors = np.column_stack((
        points[:, 0] - obs_point[0],  # x-component
        points[:, 1] - obs_point[1],  # y-component
        np.full(points.shape[0], -obs_point[2])  # z-component (negative because plane is at z=0)
    ))
    
    # Compute Euclidean distance between points and observation point
    distances = np.linalg.norm(vectors, axis=1)  # Euclidean distance

    # Compute the peak intensity
    i0 = calculate_i0_from_phi(phi, illumination)
    
    # Illuminance calculation
    if illumination == 'point':
        values = i0 / ((distances*unit)**2)
    else:
        normal_vector = np.array([0,0,1])
        cos_theta = np.abs(vectors[:, 2]) / distances  # Only the z-component is relevant for cos(theta)
        values = i0 * cos_theta / ((distances*unit)**2)
        
    
    return np.sum(values)

# Function to compute the inverse square distance from observation point to each grid point. Use 3D coordinates.
def compute_sum_at_obs_point_3d(points, obs_point, phi, unit, illumination='lambertian'):
    # check if the illumination input is correct.
    if illumination!='lambertian' and illumination!='point':
        raise ValueError("The illumination should be either lambertian or point.")
        
    # Compute the vector from each point to the observation point
    vectors = points - obs_point
    
    # Compute Euclidean distance between points and observation point
    distances = np.linalg.norm(vectors, axis=1)  # Euclidean distance

    # Compute the peak intensity
    i0 = calculate_i0_from_phi(phi, illumination)
    
    # Illuminance calculation. 
    # Note this does not assume distance = 0
    if illumination == 'point':
        values = i0 / ((distances*unit)**2)
    else:
        normal_vector = np.array([0,0,1])
        cos_theta = np.abs(vectors[:, 2]) / distances  # Only the z-component is relevant for cos(theta)
        values = i0 * cos_theta / ((distances*unit)**2)
    
    return np.sum(values)

# # Function to generate a heatmap by moving the observation point
# def generate_heatmap(width, height, dx, dy, phi, depth, unit, grid_constant=1.0, illumination='lambertian'):
#     # Generate the grid of points on the board
#     points = generate_points(width, height, dx, dy)
    
#     # Create a grid of observation points in the x-y plane
#     x_obs_range = np.linspace(-width / 2, width / 2, int(width / grid_constant))
#     y_obs_range = np.linspace(-height / 2, height / 2, int(height / grid_constant))
    
#     heatmap = np.zeros((int(height / grid_constant), int(width / grid_constant)))
    
#     # For each observation point in the x-y grid, compute the sum of values
#     for i, x_obs in enumerate(x_obs_range):
#         for j, y_obs in enumerate(y_obs_range):
#             obs_point = np.array([x_obs, y_obs, depth])
#             heatmap[j, i] = compute_sum_at_obs_point(points, obs_point, phi, unit, illumination)
    
#     return heatmap, x_obs_range, y_obs_range

# Function to generate a heatmap by moving the observation point
def generate_heatmap(width, height, dx, dy, phi, depth, unit, board_position=0.0, grid_constant=1.0, illumination='lambertian'):
    # Generate the grid of points on the board
    points = generate_points_depth(width, height, board_position, dx, dy)
    
    # Create a grid of observation points in the x-y plane
    x_obs_range = np.linspace(-width / 2, width / 2, int(width / grid_constant))
    y_obs_range = np.linspace(-height / 2, height / 2, int(height / grid_constant))
    
    heatmap = np.zeros((int(height / grid_constant), int(width / grid_constant)))
    
    # For each observation point in the x-y grid, compute the sum of values
    for i, x_obs in enumerate(x_obs_range):
        for j, y_obs in enumerate(y_obs_range):
            obs_point = np.array([x_obs, y_obs, depth])
            heatmap[j, i] = compute_sum_at_obs_point_3d(points, obs_point, phi, unit, illumination)
    
    return heatmap, x_obs_range, y_obs_range


# Precompute heatmaps for multiple depth values and store them in a numpy array
def precompute_heatmaps(width, height, dx, dy, phi, depth_range, unit, board_position=0.0, grid_constant=1.0, illumination='lambertian'):
    points = generate_points_depth(width, height, board_position, dx, dy)
    heatmaps = np.zeros((len(depth_range), int(height / grid_constant), int(width / grid_constant)))
    
    x_obs_range = np.linspace(-width / 2, width / 2, int(width / grid_constant))
    y_obs_range = np.linspace(-height / 2, height / 2, int(height / grid_constant))

    for z_index, z_obs in tqdm(list(enumerate(depth_range))):
        for i, x_obs in enumerate(x_obs_range):
            for j, y_obs in enumerate(y_obs_range):
                obs_point = np.array([x_obs, y_obs, z_obs])
                heatmaps[z_index, j, i] = compute_sum_at_obs_point_3d(points, obs_point, phi, unit, illumination)
    
    return heatmaps, x_obs_range, y_obs_range
    
# Function to plot the heatmap with adjustable vmax
def plot_heatmap(width, height, dx, dy, phi, depth, vmax, unit, board_position=0.0, grid_constant=1.0, illumination='lambertian'):
    heatmap, _, _ = generate_heatmap(
        width, height, dx, dy, phi, depth, unit, board_position=board_position, grid_constant=grid_constant, illumination=illumination
    )
    
    # Plot the heatmap using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, extent=[-width / 2, width / 2, -height / 2, height / 2],
               origin='lower', cmap='viridis', interpolation='nearest', vmin=0, vmax=vmax)
    
    plt.colorbar(label="illuminance")
    plt.title(f"Heatmap (illumination={illumination}, z={int(depth)}, phi={int(phi)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Function to plot the dual illumination heatmap with adjustable vmax
def plot_dual_heatmap(width, height, dx, dy, phi, depth, vmax, unit, board_position1=0.0, board_position2=100.0, grid_constant=1.0, illumination='lambertian'):
    heatmap1, _, _ = generate_heatmap(
        width, height, dx, dy, phi, depth, unit, board_position=board_position1, grid_constant=grid_constant, illumination=illumination
    )

    heatmap2, _, _ = generate_heatmap(
        width, height, dx, dy, phi, depth, unit, board_position=board_position2, grid_constant=grid_constant, illumination=illumination
    )

    heatmap = heatmap1+heatmap2

    # Plot the heatmap using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, extent=[-width / 2, width / 2, -height / 2, height / 2],
               origin='lower', cmap='viridis', interpolation='nearest', vmin=0, vmax=vmax)
    
    plt.colorbar(label="illuminance")
    plt.title(f"Heatmap (illumination={illumination}, z={int(depth)}, phi={int(phi)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_heatmap_precompute(heatmaps, x_obs_range, y_obs_range, vmax, depth_index):
    # Retrieve the precomputed heatmap for the current depth
    heatmap = heatmaps[depth_index]

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, extent=[x_obs_range[0], x_obs_range[-1], y_obs_range[0], y_obs_range[-1]],
               origin='lower', cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(label="illuminance")
    plt.title(f"Heatmap (z={int(depth)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()