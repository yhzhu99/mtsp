import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import colors as mcolors

# matplotlib_colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

matplotlib_colors = [
    "black",
    "red",
    "yellow",
    "grey",
    "brown",
    "darkred",
    "peru",
    "darkorange",
    "darkkhaki",
    "steelblue",
    "blue",
    "cyan",
    "green",
    "navajowhite",
    "lightgrey",
    "lightcoral",
    "mediumblue",
    "midnightblue",
    "blueviolet",
    "violet",
    "fuchsia",
    "mediumvioletred",
    "hotpink",
    "crimson",
    "lightpink",
    "slategray",
    "lime",
    "springgreen",
    "teal",
    "beige",
    "olive",
]


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if np.array_equal(value, item_to_find):
            indices.append(idx)
    return indices


def plot_results(Best_path, iterations, best_record):
    # print(find_indices(Best_path, [0, 0]))

    # Best_path = np.vstack([Best_path, Best_path[0]])
    # Best_path = np.vstack([Best_path[0], Best_path])
    # print(Best_path[0], Best_path[-1])

    if not np.array_equal(Best_path[0], [0, 0]):
        Best_path = np.vstack([[0, 0], Best_path])
    if not np.array_equal(Best_path[-1], [0, 0]):
        Best_path = np.vstack([Best_path, [0, 0]])
    # print(Best_path)

    found_start_points_indices = find_indices(Best_path, [0, 0])
    result_paths = []

    for j in range(len(found_start_points_indices) - 1):
        from_index = found_start_points_indices[j]
        end_index = found_start_points_indices[j + 1]
        path = []
        for k in range(from_index, end_index + 1):
            path.append(Best_path[k])
        path = np.array(path)
        result_paths.append(path)

    # print(Best_path)
    # print(result_paths)

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
    axs[0].scatter(Best_path[:, 0], Best_path[:, 1])

    for ix, path in enumerate(result_paths):
        axs[0].plot(path[:, 0], path[:, 1], color=matplotlib_colors[ix], alpha=0.8)
    # axs[0].plot(Best_path[:, 0], Best_path[:, 1], color="green", alpha=0.1)

    # Draw start point
    axs[0].plot([0], [0], marker="*", markersize=20, color="red")

    axs[0].set_title("Searched Best Solution")

    axs[1].plot(iterations, best_record)
    axs[1].set_title("Convergence Curve")
    plt.show()
