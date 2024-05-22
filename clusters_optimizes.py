import numpy as np
from jenkspy import JenksNaturalBreaks
import math
import random
import matplotlib.pyplot as plt


def adjust_y_position(current_y, previous_positions, min_distance=1500, delta=1500):
    """
    Adjusts the current y-position to avoid overlap with previous positions.

    :param current_y: The initial y-position proposed for the current label.
    :param previous_positions: List of y-positions already used for other labels.
    :param min_distance: The minimum vertical distance to maintain between labels.
    :return: An adjusted y-position that does not overlap with previous positions.
    """
    # Sort the list of previous positions to maintain order
    previous_positions.sort()

    # Check if the current position overlaps with any previous position
    while any(abs(current_y - prev_y) < min_distance for prev_y in previous_positions):
        current_y += delta  # Move the current y-position up by the minimum distance

    # Append the adjusted position to the list of used positions
    # previous_positions.append(current_y)

    return current_y


def calculate_vertical_offset(mean, upper_bound, scale_factor=0.05):
    """
    Calculates a vertical offset for annotations based on the mean and upper bound.

    :param mean: The mean value around which the error bar is centered.
    :param upper_bound: The upper error bound from the mean.
    :param scale_factor: A scaling factor to determine the offset based on the range.
                         Default is 0.05 (5% of the distance from mean to upper bound).
    :return: The calculated vertical offset.
    """
    # Calculate the distance from the mean to the upper bound
    distance = upper_bound - mean

    # Calculate offset as a scaled percentage of the distance
    offset = distance * scale_factor

    # Ensure there's a minimal absolute offset in cases where distance is very small
    min_offset = mean * 0.02  # Minimum offset as 2% of the mean, adjust as needed
    return max(offset, min_offset)


def plot_clusters(sorted_losses_by_control_clustered, exp, number, x, y):
    labels = [', '.join(map(str, ctrl)) for ctrl, *_ in sorted_losses_by_control_clustered]
    means = [mean for _, mean, *_ in sorted_losses_by_control_clustered]
    stds = [data[2] if len(data) > 2 else None for data in sorted_losses_by_control_clustered]
    errors = [[data[2] * data[3][0], data[2] * data[3][1]] if len(data) > 2 else [0, 0] for data in
              sorted_losses_by_control_clustered]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 14), dpi=60)

    # Error bar plot
    x_positions = np.arange(len(labels))
    prev_y = []

    # # Draw error bars and mean points
    for i, (mean, error, std) in enumerate(zip(means, errors, stds)):
        if std is not None:  # Draw error bars if std is present
            # Lower error
            ax.plot([x_positions[i], x_positions[i]], [mean, max(mean - error[0], 0)], color='red', marker='', linestyle='-')
            ax.plot([x_positions[i] - 0.05, x_positions[i] + 0.05], [max(mean - error[0], 0), max(mean - error[0], 0)],
                    color='red')  # Cap

            # Upper error
            ax.plot([x_positions[i], x_positions[i]], [mean, mean + error[1]], color='green', marker='', linestyle='-')
            ax.plot([x_positions[i] - 0.05, x_positions[i] + 0.05], [mean + error[1], mean + error[1]],
                    color='green')  # Cap

        ax.plot(x_positions[i], mean, 'o', color='dodgerblue')  # Just the mean point

        # Annotation for mean and std values
        if std:
            annotation_text = f'Mean: ${mean:,.0f}\nStd: ${std:,.0f}'
        else:
            if exp:
                annotation_text = f'Mean: ${mean:,.0f}'
            else:
                annotation_text = f'Loss: ${mean:,.0f}'

        # Dynamic positioning to avoid overlaps
        vertical_offset = 20000 # Adjust based on your data scale
        vertical_offset = calculate_vertical_offset(mean, mean + error[1], scale_factor=0.05)
        text_y_position = mean + (error[1] if std else 0) + vertical_offset
        text_y_position = adjust_y_position(text_y_position, prev_y)
        prev_y.append(text_y_position)
        le = error[0]
        ue = error[1]
        right_offset = 0.2
        if std is not None:
            ax.text(x_positions[i] + right_offset, mean - max(le, 2000), f'${max(mean - error[0], 0):,.0f}', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='left', color='red')
            ax.text(x_positions[i] + right_offset, mean + ue, f'${mean + ue:,.0f}', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='left', color='green')

        # Annotations with connecting lines
        ax.annotate(annotation_text, xy=(x_positions[i], mean), xytext=(x_positions[i], text_y_position),
                    textcoords="data", ha='center', va='bottom', fontsize=16,
                    # arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='blue')
                    )

    # Aesthetic adjustments for a professional look
    if exp:
        ax.set_xlabel('Control Numbers', fontsize=16, labelpad=10)
        ax.set_ylabel('Mean Loss', fontsize=16, labelpad=10)
        ax.set_title(f'Mean Loss by Control Numbers with Clusters, Experiment #{number+1}', fontsize=14, pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)

        # Adding grid for better readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout to avoid clipping of tick-labels
        plt.tight_layout()
        plt.savefig(f'plots/exp{number}Clusteredx={x},y={y}.png')

    else:
        ax.set_xlabel('Control Numbers', fontsize=16, labelpad=10)
        ax.set_ylabel('Actual Loss', fontsize=16, labelpad=10)
        ax.set_title(f'Actual Loss by Control Numbers without Clusters, Experiment #{number+1}', fontsize=14, pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)

        # Adding grid for better readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout to avoid clipping of tick-labels
        plt.tight_layout()
        plt.savefig(f'plots/exp{number}Original.png')
    # Show the plot
    plt.show()

def transform_true_losses(true_losses):
    new = []
    for control, loss in true_losses:
        new.append(((control,), loss))
    return new



def loss_per_control_failure(losses_array, user_index_control_number_map):
    """
    Given losses array which is the total loss per participant
    and user index control map which maps user index id to the
    controls they reported as failed, this function will return
    the total loss per control
    """
    control_loss_map = dict()
    for user_index, loss in enumerate(losses_array):
        user_failed_controls = user_index_control_number_map[user_index]
        user_base_number = loss / len(user_failed_controls)
        for control in user_failed_controls:
            if control not in control_loss_map:
                control_loss_map[control] = 0
            control_loss_map[control] += user_base_number
    return control_loss_map


def select_optimal_clustering(clusters_dict, weight_inter_distance=0.33, weight_intra_diversity=0.33, weight_num_groups=0.33):
    """
    Given the clusters dict and the weights for parameters (inter distance, intra distance,
    and cluster counts, this function returns the highest scoring cluster.
    :param clusters_dict:
    :param weight_inter_distance: the weight given for the inter-diversity
    :param weight_intra_diversity:
    :param weight_num_groups:
    :return:
    """
    best_score = float('-inf')
    best_cluster_config = None

    for num_clusters, cluster_info in clusters_dict.items():
        clusters = cluster_info['Clusters']
        var_within = sum(list(cluster_info["Variance-within-clusters"].values()))/ len(clusters)
        var_between = cluster_info["Variance-between-clusters"]
        # criterion 1: minimize average inter-cluster distance
        avg_inter_cluster_distance = sum(cluster_info['Inter-cluster Distances']) / len(cluster_info['Inter-cluster Distances'])
        # criterion 2: maximize average intra-cluster diversity
        avg_intra_cluster_diversity = sum(cluster_info['Intra-cluster Diversity'].values()) / len(cluster_info['Intra-cluster Diversity'])
        # criterion 3: more groups are better
        cluster_count = num_clusters

        print(f"var within is: {var_within} while intra is: {avg_intra_cluster_diversity}\n"
              f"var between is: {var_between} while inter is: {avg_inter_cluster_distance}\n")

        # Create a weighted score
        score = (weight_inter_distance * (1 / avg_inter_cluster_distance) +
                 weight_intra_diversity * avg_intra_cluster_diversity +
                 weight_num_groups * cluster_count)

        # Select the configuration with the highest score
        if score > best_score:
            best_score = score
            best_cluster_config = (num_clusters, cluster_info)

    return best_cluster_config[1]


def evaluate_clusters(loss_per_control, x, y):
    """
    Given x,y and the loss per control dict <control>:<loss>
    This function will run the Jenks Natural Break algorithm from x clusters
    to len(loss_per_control) clusters (each point is a cluster) and will return the
    cluster structures which have at least x clusters with y elements each

    The organized_data dict will be as the following:
    <number of clusters> : {
                "Inter-cluster Distances": list of range of each cluster,
                "Intra-cluster Diversity": list of std of each cluster,
                "Variance-within-clusters": avg_variance_within_clusters_correct(clusters),
                "Variance-between-clusters": variance_between_clusters_correct(clusters),
                "Breaks": breaks,
                "Labels": labels,
                "Clusters": {k: [v[0] for v in vals] for k, vals in clusters.items()}  # Storing control keys
            }
    <number of
    """
    organized_data = {}
    # Prepare data for clustering, retaining original keys
    data = list(loss_per_control.items())
    values = np.array([item[1] for item in data])  # Extracting the loss values for clustering
    for num_clusters in range(x, len(set(values)) + 1):  # Explore up to the number of unique values
        jnb = JenksNaturalBreaks(num_clusters)
        jnb.fit(values)
        labels = jnb.labels_
        breaks = jnb.breaks_
        # Create clusters based on labels
        clusters = {i: [] for i in range(num_clusters)}
        for (control, value), label in zip(data, labels):
            clusters[label].append((control, value))
        # Clusters = {jnb label: [(<control #, total_loss)]} -> {0: [(21, 10600.0), (2, 10600.0), (6, 18666.666666666668), (12, 19000.0)]
        # Check each cluster for at least y unique elements
        # sufficient_size = sum(len(set(val for _, val in clusters[i])) >= y for i in clusters)
        sufficient_size = 0
        for cluster_number, points in clusters.items():
            losses_set = set()
            for (control_number, loss) in points:
                losses_set.add(loss)
            if len(losses_set) >= y:
                sufficient_size += 1
        # sufficient_size = how many clusters have at least y elements

        if sufficient_size >= x:
            distances = inter_cluster_distance(breaks) # range of cluster
            diversity = intra_cluster_diversity(clusters) # std of cluster
            organized_data[num_clusters] = {
                "Inter-cluster Distances": distances,
                "Intra-cluster Diversity": diversity,
                "Variance-within-clusters": avg_variance_within_clusters_correct(clusters),
                "Variance-between-clusters": variance_between_clusters_correct(clusters),
                "Breaks": breaks,
                "Labels": labels,
                "Clusters": {k: [v[0] for v in vals] for k, vals in clusters.items()}  # Storing control keys
            }
    if not organized_data:
        raise ValueError(f"Cannot construct at least {x} clusters with at least {y} elements each.\n"
                         f"Try different numbers")
    return organized_data


def avg_variance_within_clusters_correct(clusters):
    """
    clusters:
    {0: [(15, 11000.0), (12, 14250.0), (19, 13666.666666666666), (17, 20500.0), (9, 20500.0), (13, 17600.0)],
    1: [(20, 31266.666666666664), (8, 38100.0), (5, 29500.0)],
    2: [(11, 54750.0), (0, 64250.0)],
    3: [(6, 106250.0), (21, 95500.0)]}
    Basically, for each of the clusters, this will calculate the variance within the values
    of each cluster  and then will return the average of the calculated variances.
    For the example above, this will calculate 4 variances and then will return the avg
    """
    variance_within = {}
    for cluster_id, loss_per_control in clusters.items():
        # loss per control is an array:  [(21, 10600.0), (2, 10600.0), (6, 18666.666666666668), (12, 19000.0)]
        values = [loss_pair[1] for loss_pair in loss_per_control]
        if len(values) > 1:
            variance_within[cluster_id] = np.var(values, ddof=1)  # ddof=1 for sample variance
        else:
            variance_within[cluster_id] = 0  # variance is zero if there is only one element
    # variance_within = {0: 44327358.58585859, 1: 20629259.25925926, 2: 28349537.03703703, 3: 57781250.0}
    if variance_within:
        return sum(list(variance_within.values()))/len(variance_within)
    return 0


def variance_between_clusters_correct(clusters):
    """
    clusters:
    {0: [(15, 11000.0), (12, 14250.0), (19, 13666.666666666666), (17, 20500.0), (9, 20500.0), (13, 17600.0)],
    1: [(20, 31266.666666666664), (8, 38100.0), (5, 29500.0)],
    2: [(11, 54750.0), (0, 64250.0)],
    3: [(6, 106250.0), (21, 95500.0)]}
    """
    # calculate overall mean of all data points
    all_values = [loss_pair[1] for cluster_losses in clusters.values() for loss_pair in cluster_losses]
    overall_mean = np.mean(all_values)

    # calculate means of each cluster
    cluster_means = []
    for controls_and_losses in clusters.values():
        if controls_and_losses:  # ensure there are controls in the cluster to avoid empty data
            values = [loss for (_, loss) in controls_and_losses]
            cluster_means.append(np.mean(values))

    if cluster_means:
        # Calculate the squared differences from the overall mean and then their average (variance)
        mean_deviations = [(cluster_mean - overall_mean) ** 2 for cluster_mean in cluster_means]
        n = len(cluster_means)
        if n > 1:
            # sample variance
            n -= 1
        variance_between = np.sum(mean_deviations) / n
    else:
        variance_between = 0  # if no clusters or one cluster, variance is zero or undefined
    # 'var_between': 1927401969.2930992
    return variance_between if variance_between else 0


def inter_cluster_distance(breaks):
    # calculates the range of each cluster
    distances = [j - i for i, j in zip(breaks[:-1], breaks[1:])]
    return distances

def intra_cluster_diversity(clusters):
    # calculated the std within each cluster
    diversity = {k: np.std([v[1] for v in vals]) for k, vals in clusters.items()}
    return diversity


def process_cluster_results(cluster_info, loss_per_control, min_std=5000, min_band=1):
    results = {}
    for cluster_id, controls in cluster_info['Clusters'].items():
        # Extract values using control keys to calculate statistics
        values = [loss_per_control[control] for control in controls]

        # Sort controls by their corresponding loss values
        sorted_controls = sorted(controls, key=lambda control: loss_per_control[control])

        # Calculate statistical metrics
        values_array = np.array(values)
        mean_val = max(round(np.mean(values_array), -3), min_std)
        print(mean_val)
        std_val = max(round(np.std(values_array), -3), min_std)
        min_val = round(np.min(values_array), -3)
        max_val = round(np.max(values_array), -3)

        # Calculate ceiling of std range from min to max
        # Calculate the number of standard deviations from the mean for min and max values
        if std_val > 0:  # To avoid division by zero
            min_std_from_mean = max(math.ceil((mean_val - min_val) / std_val), min_band)
            max_std_from_mean = max(math.ceil((max_val - mean_val) / std_val), min_band)
            std_range = (min_std_from_mean, max_std_from_mean)
        else:
            std_range = (0, 0)

        results[cluster_id] = {
            'Sorted Controls': sorted_controls,
            'Mean': mean_val,
            'Standard Deviation': std_val,
            'STD Range Ceiling': std_range,
            'Min': min_val,
            'Max': max_val
        }
        # if len(sorted_controls) ==1:
        #     results[cluster_id]["Value"] = mean_val

    return results


def extract_metrics(clusters_dict):
    results = {}
    for num_clusters, cluster_info in clusters_dict.items():
        results[num_clusters] = {
            'var_within': (num_clusters, cluster_info['Variance-within-clusters']),
            'var_between': (num_clusters, cluster_info['Variance-between-clusters']),
            'num_clusters': (num_clusters, num_clusters)
        }
    return results




def rank_clusters(metrics):
    # Convert each metric to a numpy array for argsort operation
    var_within = np.array([info['var_within'] for info in metrics.values()])
    var_between = np.array([info['var_between'] for info in metrics.values()])
    num_clusters = np.array([info['num_clusters'] for info in metrics.values()])

    # Ranks (argsort returns indices that would sort an array, [::-1] for descending order where higher is better)
    ranks = {
        'var_within': var_within.argsort().argsort(),  # Lower is better
        'var_between': (-var_between).argsort().argsort(),  # Higher is better
        'num_clusters': (-num_clusters).argsort().argsort()  # Adjust according to preference
    }
    return ranks


def score_clusters(ranks, weights):
    # ranks = {'var_within': [(0, 1), (1, 2)], 'var_between': [(0, 2), (1, 1)], 'num_clusters': [(0, 2), (1, 1)]}
    #                          <rank, cluster numbers>
    # my goal is to have <score, cluster number>
    scores = {}
    for rank_name, values in ranks.items():
        for (rank, label) in values:
            if scores.get(rank_name) is None:
                scores[rank_name] = []
            scores[rank_name].append(
                ((len(ranks[rank_name])- rank) * weights[rank_name], label)
            )
    # scores = {'var_within': [(0.4, 1)], 'var_between': [(0.4, 1)], 'num_clusters': [(0.2, 1)]}
    return scores


def find_winner_index(scores):
    # return a list of tuples reverse sorted
    # [(<cluster number>, <score value>)]
    label_score_dict = dict()
    for score_lists in scores.values():
        for score, cluster_num in score_lists:
            if cluster_num not in label_score_dict:
                label_score_dict[cluster_num] = 0
            label_score_dict[cluster_num] += score
    label_scores_tup_list = []
    for cluster_num, value in label_score_dict.items():
        label_scores_tup_list.append((cluster_num, value))
    return sorted(label_scores_tup_list, key=lambda x: x[1], reverse=True)


def rank_clusters_correct(metrics):
    # Prepare arrays to hold average variance within, variance between, and number of clusters
    # basically takes all the different clusters constructed and arranged each
    # one of our cluster analysis params in a sorted list
    # cluster analysis params == variaince within clusters, variance between clusters, and total number of clusters
    var_within_avgs = []
    var_between = []
    num_clusters = []
    # Fill arrays
    for i, (cluster_numbers_key ,data) in enumerate(metrics.items()):
        # Calculate average variance within clusters for this configuration
        var_within_avgs.append((cluster_numbers_key, data['var_within']))
        var_between.append((cluster_numbers_key, data['var_between']))
        num_clusters.append((cluster_numbers_key, data['num_clusters']))
    # Ranks (argsort returns indices that would sort an array, [::-1] for descending order where higher is better)

    within_ranking = sorted(var_within_avgs, key=lambda x: x[1]) # lower is better, 0 being highest
    within_ranking = [(rank, key) for rank, (key, _) in enumerate(within_ranking)]
    between_ranking = sorted(var_between, key=lambda x: x[1], reverse=True) #higher is better, descending order, rank is index
    between_ranking = [(rank, key) for rank, (key, _) in enumerate(between_ranking)]
    cluster_ranking = sorted(num_clusters, key=lambda x: x[1], reverse=True)  # more clusters should be higher
    cluster_ranking = [(rank, key) for rank, (key, _) in enumerate(cluster_ranking)]
    ranks = {
        'var_within':within_ranking,
        'var_between': between_ranking,
        'num_clusters': cluster_ranking,
    }

    return ranks


def generate_loss(distribution):
    threshold = random.random()
    cumulative_probability = 0.0
    for range_limits, probability in distribution.items():
        cumulative_probability += probability
        if threshold < cumulative_probability:
            return random.randint(*range_limits)


def generate_data(distribution_name, companies_reported):
    if distribution_name not in DISTRIBUTIONS:
        raise ValueError('Invalid distribution')
    losses_array = []
    user_index_control_number_map = dict()
    for company in range(companies_reported):
        # need to choose loss
        loss = generate_loss(DISTRIBUTIONS[distribution_name])*1000
        losses_array.append(loss)
        number_of_controls_reported = random.randint(*CONTROLS_REPORTED_RANGE)
        controls_reported = random.sample(range(NUMBER_OF_CONTROLS), number_of_controls_reported)
        user_index_control_number_map[company] = controls_reported
    return losses_array, user_index_control_number_map


NUMBER_OF_CONTROLS = 12
CONTROLS_REPORTED_RANGE = (1, 5)
DISTRIBUTIONS = {
    'outliers': {(5, 100): 0.75, (500, 1000): 0.25},
    "spread": {(10, 30): 0.33, (60, 90): 0.33, (100, 150): 0.33},
    'regular': {(5, 100): 1}
}
WEIGHTS = {'var_within': 0.4, 'var_between': 0.4, 'num_clusters': 0.2}

losses_array, user_index_control_number_map = generate_data("regular", 3)
"""
The data looks like the following:

losses_array = [23058.0, 500000.0, 72650.0, 32500.0]
count_controls_reported = [2, 5, 5, 2]
user_index_control_number_map = {
    0: ["1a", "2a"],
    1: ["2b", "5a", "5b", "8a", "8b"],
    2: ["5b", "6b", "8a", "8b", "8c"],
    3: ["5a", "5b"]
}
"""

x = 2
y = 2
loss_per_control = loss_per_control_failure(losses_array, user_index_control_number_map)
# loss per control = {<control number> : <total loss>}
valid_clusters = evaluate_clusters(loss_per_control, x, y)  # Call with minimum clusters and minimum unique elements per cluster
print("=======VALID CLUSTERS=======")
print(valid_clusters)
metrics = extract_metrics(valid_clusters) # just organizes the data
# metrics: key=num_clusters, value={var_within: <>, var_between: <>, num_clusters:<>}
# it will have for all of the different valid x,y clusters
ranks = rank_clusters_correct(metrics)
# print(ranks)
overall_scores = score_clusters(ranks, WEIGHTS)
print(f"Scores: {overall_scores}")
sorted_scoring_list = find_winner_index(overall_scores)
highest_scoring_cluster = sorted_scoring_list[0][0]
optimal_cluster = valid_clusters[highest_scoring_cluster]
print(f"Optimal: {optimal_cluster}")
output = process_cluster_results(optimal_cluster, loss_per_control, min_std=5000, min_band=1)
print(output)
clustered_ans = []
for cluster_id, cluster_info in output.items():
    """
     output[cluster_id] = {
            'Sorted Controls': sorted_controls,
            'Mean': mean_val,
            'Standard Deviation': std_val,
            'STD Range Ceiling': std_range,
            'Min': min_val,
            'Max': max_val
        }
    """
    controls = tuple(cluster_info["Sorted Controls"])
    clustered_ans.append((
        controls,
        cluster_info["Mean"],
        cluster_info["Standard Deviation"],
        cluster_info["STD Range Ceiling"]))
print(clustered_ans)

######## graphing
plot_clusters(clustered_ans, exp=True, number=0, x=x, y=y)
