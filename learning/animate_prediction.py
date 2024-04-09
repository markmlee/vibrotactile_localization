import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_height_rad():
    # Cylinder parameters
    radius = 9
    height = 20
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, height, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    
    data_path_gt = 'y_val.npy'
    data_path_pred = 'y_pred.npy'
    #load data from .npy file
    y_val = np.load(data_path_gt)
    y_pred = np.load(data_path_pred)


    data = y_val
    estimate = y_pred

    # Initialize figure and 3D subplots
    fig = plt.figure(figsize=(15, 5))

    # Isometric view
    ax_iso = fig.add_subplot(131, projection='3d')
    ax_iso.view_init(elev=30, azim=45)

    # Top-down view
    ax_top = fig.add_subplot(132, projection='3d')
    ax_top.view_init(elev=90, azim=0)

    # Side view
    ax_side = fig.add_subplot(133, projection='3d')
    ax_side.view_init(elev=0, azim=0)

    # Draw cylinders in each subplot
    for ax in [ax_iso, ax_top, ax_side]:
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='blue')
        #add subtitle according to each view
        ax.set_title('Isometric view') if ax == ax_iso else ax.set_title('Top-down view') if ax == ax_top else ax.set_title('Side view')


    # initialize pts for ground truth and prediction
    point_iso_gt, = ax_iso.plot([], [], [], 'ro')
    point_top_gt, = ax_top.plot([], [], [], 'ro')
    point_side_gt, = ax_side.plot([], [], [], 'ro')

    point_iso_pred, = ax_iso.plot([], [], [], 'bx')
    point_top_pred, = ax_top.plot([], [], [], 'bx')
    point_side_pred, = ax_side.plot([], [], [], 'bx')

    def animate(i):
        # Update ground truth and prediction
        h, angle = data[i % len(data)]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = h

        h_pred, angle_pred = estimate[i % len(estimate)]
        x_pred = radius * np.cos(angle_pred)
        y_pred = radius * np.sin(angle_pred)
        z_pred = h_pred

        
        # Update each point for ground truth
        for point, ax in zip([point_iso_gt, point_top_gt, point_side_gt], [ax_iso, ax_top, ax_side]):
            point.set_data([x], [y])
            point.set_3d_properties([z])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

        # Update each point for prediction
        for point, ax in zip([point_iso_pred, point_top_pred, point_side_pred], [ax_iso, ax_top, ax_side]):
            point.set_data([x_pred], [y_pred])
            point.set_3d_properties([z_pred])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

        return point_iso_gt, point_top_gt, point_side_gt, point_iso_pred, point_top_pred, point_side_pred

    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(data), interval=1000, blit=True)

    # Save animation as GIF
    ani.save('cylinder_views.gif', writer='imagemagick')

    plt.show()

def animate_xy(y_gt, y_pred):
    """
    y = [height,x,y]
    input: y ground truth, and y prediction
    output: animation of both ground truth and prediction in 3D plot
    
    animate x,y,height 3D plot
    """
    # Cylinder parameters
    radius = 9
    height = 20
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, height, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)


    data = y_gt
    estimate = y_pred

    print(f"dimensions of data: {data.shape}, estimate: {estimate.shape}")

    # Initialize figure and 3D subplots
    fig = plt.figure(figsize=(15, 5))

    # Isometric view
    ax_iso = fig.add_subplot(131, projection='3d')
    ax_iso.view_init(elev=30, azim=45)

    # Top-down view
    ax_top = fig.add_subplot(132, projection='3d')
    ax_top.view_init(elev=90, azim=0)

    # Side view
    ax_side = fig.add_subplot(133, projection='3d')
    ax_side.view_init(elev=0, azim=0)

    # Draw cylinders in each subplot
    for ax in [ax_iso, ax_top, ax_side]:
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='blue')
        #add subtitle according to each view
        ax.set_title('Isometric view') if ax == ax_iso else ax.set_title('Top-down view') if ax == ax_top else ax.set_title('Side view')


    # initialize pts for ground truth and prediction
    point_iso_gt, = ax_iso.plot([], [], [], 'ro')
    point_top_gt, = ax_top.plot([], [], [], 'ro')
    point_side_gt, = ax_side.plot([], [], [], 'ro')

    point_iso_pred, = ax_iso.plot([], [], [], 'bx')
    point_top_pred, = ax_top.plot([], [], [], 'bx')
    point_side_pred, = ax_side.plot([], [], [], 'bx')

    def animate(i):
        # Update ground truth and prediction
        h = data[i % len(data)][0]
        x = data[i % len(data)][1] * radius
        y = data[i % len(data)][2] * radius
        z = h



        h_pred = estimate[i % len(estimate)][0]
        x_pred = estimate[i % len(estimate)][1] * radius
        y_pred = estimate[i % len(estimate)][2] * radius

        #project x,y to unit circle
        # radian = np.arctan2(y_pred,x_pred)
        # x_pred = radius * np.cos(radian)
        # y_pred = radius * np.sin(radian)
        z_pred = h_pred

        
        # Update each point for ground truth
        for point, ax in zip([point_iso_gt, point_top_gt, point_side_gt], [ax_iso, ax_top, ax_side]):
            point.set_data([x], [y])
            point.set_3d_properties([z])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

        # Update each point for prediction
        for point, ax in zip([point_iso_pred, point_top_pred, point_side_pred], [ax_iso, ax_top, ax_side]):
            point.set_data([x_pred], [y_pred])
            point.set_3d_properties([z_pred])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

        return point_iso_gt, point_top_gt, point_side_gt, point_iso_pred, point_top_pred, point_side_pred

    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(data), interval=1000, blit=True)

    # Save animation as GIF
    ani.save('cylinder_views.gif', writer='imagemagick')

    plt.show()



def main():
    data_path_gt = 'y_val.npy'
    data_path_pred = 'y_pred.npy'

    #load data from .npy file
    y_val = np.load(data_path_gt)
    y_pred = np.load(data_path_pred)

    #downsample 1/10 of the data
    y_val = y_val[::10]
    y_pred = y_pred[::10]

    animate_xy(y_val, y_pred)

if __name__ == '__main__':
    main()