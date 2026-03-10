import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, List, TYPE_CHECKING
from consts import Players, ObservationConsts, ValueFunctionType
from models import get_discounted_future_rewards_at_every_step
import os
import numpy as np

if TYPE_CHECKING:
    from models import Trajectory, ValueFunction

# Plotting actions over the trajectory
def _plot_player_positions(ax: Axes, trajectory: 'Trajectory', orientation=str) -> Axes:
    assert orientation in ["horizontal", "vertical"]

    player_positions: Dict[Players, List[float]] = {
        Players.HUMAN: [step.state.players_state.human_position_x for step in trajectory],
        Players.SHUTTER: [step.state.players_state.shutter_position_x for step in trajectory],
        Players.NAO: [step.state.players_state.nao_position_x for step in trajectory]
    }

    if orientation == "horizontal":
        for player, positions in player_positions.items():
            ax.plot(range(len(positions)), positions, label=f'{player.value} X Position')
        ax.set_title('Player X Positions Over Trajectory (Horizontal)')
        ax.set_xlabel('Trajectory Step Index')
        ax.set_ylabel('X Position')
        ax.invert_yaxis()  # Invert y axis to match game orientation
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax.legend()
        return ax
    elif orientation == "vertical":
        for player, positions in player_positions.items():
            ax.plot(positions, range(len(positions)), label=f'{player.value} X Position')
        ax.set_title('Player X Positions Over Trajectory (Vertical)')
        ax.set_ylabel('Trajectory Step Index')
        ax.set_xlabel('X Position')
        ax.invert_yaxis()
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax.legend()

def _plot_player_actions(ax: Axes, trajectory: 'Trajectory') -> Axes:
    # action_labels = [step.action.short_str() for step in trajectory]
    # unique_action_labels = list(set(action_labels))
    # CUSTOM_ACTION_ORDER = ["Action()", "Action(left)", "Action(right)", "Action(shoot)"]
    # ordered_unique_actions = CUSTOM_ACTION_ORDER + sorted([action_label for action_label in unique_action_labels if action_label not in CUSTOM_ACTION_ORDER])
    # action_indices_remapped = [ordered_unique_actions.index(action_label) for action_label in action_labels]

    CUSTOM_ACTION_ORDER = ["Action()", "Action(left)", "Action(right)", "Action(shoot)"]
    # player_enacted_action_indices = {
    #     Players.HUMAN: [step.info.actions_taken[Players.HUMAN].index() for step in trajectory],
    #     Players.SHUTTER: [step.info.actions_taken[Players.SHUTTER].index() for step in trajectory],
    #     Players.NAO: [step.info.actions_taken[Players.NAO].index() for step in trajectory]
    # }

    player_enacted_action_labels = {
        Players.HUMAN: [step.info.actions_taken[Players.HUMAN].short_str() for step in trajectory],
        Players.SHUTTER: [step.info.actions_taken[Players.SHUTTER].short_str() for step in trajectory],
        Players.NAO: [step.info.actions_taken[Players.NAO].short_str() for step in trajectory]
    }
    player_enacted_action_indices_remapped = {
        player: [CUSTOM_ACTION_ORDER.index(action_label) for action_label in actions]
        for player, actions in player_enacted_action_labels.items()
    }

    # Subplot for actions over trajectory
    # ax2 = plt.subplot2grid((5, 2), (1, 0))
    # ax2.scatter(range(len(action_indices_remapped)), action_indices_remapped)
    # ax2.set_title('Actions Over Trajectory')
    # ax2.set_xlabel('Trajectory Step Index')
    # ax2.set_ylabel('Action Index')
    # ax2.set_yticks(range(len(ordered_unique_actions)))
    # ax2.set_yticklabels(ordered_unique_actions)
    # ax2.grid(True)

    for player, indices in player_enacted_action_indices_remapped.items():
        ax.scatter(range(len(indices)), indices, label=player.value, alpha=0.5)
    ax.set_title('Actions Taken by All Players Over Trajectory')
    ax.set_xlabel('Trajectory Step Index')
    ax.set_ylabel('Action Index')
    ax.set_yticks(range(len(CUSTOM_ACTION_ORDER)))
    ax.set_yticklabels(CUSTOM_ACTION_ORDER)
    ax.grid(True)
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    ax.legend()
    return ax

def _plot_nao_support(ax: Axes, trajectory: 'Trajectory') -> Axes:
    # Assuming `trajectory` has a `nao_support` attribute that is a list of NAO support values
    nao_support = []
    player_to_plot_idx = {
        Players.HUMAN: 1,
        Players.SHUTTER: 0,
        Players.NAO: -1,
    }
    plot_idx_to_player = {
        idx: player for player, idx in player_to_plot_idx.items()
    }
    # previous_support_counts = {Players.HUMAN: 0, Players.SHUTTER: 0}
    # current_support = None
    # for i, step in enumerate(trajectory):
    #     if i == 0:
    #         # TODO: log support another way to get it for step 0?
    #         continue
    #     support_counts = step.state.history.support_frame_count
    #     if support_counts[Players.HUMAN] > previous_support_counts[Players.HUMAN]:
    #         assert support_counts[Players.SHUTTER] == previous_support_counts[Players.SHUTTER]
    #         current_support = Players.HUMAN
    #     elif support_counts[Players.SHUTTER] > previous_support_counts[Players.SHUTTER]:
    #         assert support_counts[Players.HUMAN] == previous_support_counts[Players.HUMAN]
    #         current_support = Players.SHUTTER
    #     nao_support.append((i, player_to_plot_idx[current_support]))
    #     previous_support_counts = support_counts

    for i, step in enumerate(trajectory):
        #nao_support.append((i, step.state.players_state.nao_support))
        current_support = step.state.players_state.nao_supporting_player
        nao_support.append((i, player_to_plot_idx[current_support]))


    step_indices, support_plot_indices = zip(*nao_support)
    ax.scatter(step_indices, support_plot_indices)
    ax.set_yticks(sorted(player_to_plot_idx.values()))
    ax.set_yticklabels([plot_idx_to_player[idx].value for idx in sorted(player_to_plot_idx.values())])

    ax.set_title('NAO Support Over Trajectory')
    ax.set_xlabel('Trajectory Step Index')
    ax.set_ylabel('NAO Support')
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    return ax

def _plot_events(ax: Axes, trajectory: 'Trajectory') -> Axes:
    from consts import RewardConsts
    events = []
    for i, step in enumerate(trajectory):
        for player in step.info.players_hit:
            events.append((i, f"{player.value} Hit"))
        for player in step.info.players_respawned:
            events.append((i, f"{player.value} Respawned"))
        player_taking_score_lead_rewards = step.info.player_taking_score_lead_rewards
        new_leader = step.state.get_leading_score_player(threshold=RewardConsts.LEADING_SCORE_THRESHOLD)
        if player_taking_score_lead_rewards[Players.HUMAN] != 0 and player_taking_score_lead_rewards[Players.SHUTTER] != 0:
            # Some leader change event occurred
            if new_leader == Players.HUMAN:
                events.append((i, f"{Players.HUMAN.value} Takes Score Lead"))
            elif new_leader == Players.SHUTTER:
                events.append((i, f"{Players.SHUTTER.value} Takes Score Lead"))
            elif new_leader == None:
                if player_taking_score_lead_rewards[Players.HUMAN] < player_taking_score_lead_rewards[Players.SHUTTER]:
                    events.append((i, f"{Players.HUMAN.value} Loses Score Lead"))
                elif player_taking_score_lead_rewards[Players.HUMAN] > player_taking_score_lead_rewards[Players.SHUTTER]:
                    events.append((i, f"{Players.SHUTTER.value} Loses Score Lead"))

    unique_event_labels = sorted({event[1] for event in events})
    
    event_indices = [event[0] for event in events]
    event_labels = [event[1] for event in events]
    ax.scatter(event_indices, [unique_event_labels.index(label) for label in event_labels])
    ax.set_title('Events Over Trajectory')
    ax.set_xlabel('Trajectory Step Index')
    ax.set_ylabel('Event')
    ax.set_yticks(range(len(unique_event_labels)))
    ax.set_yticklabels(unique_event_labels)
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    return ax

def _plot_player_scores(ax: Axes, trajectory: 'Trajectory') -> Axes:
    player_scores = {
        Players.HUMAN.value: [step.state.score_state.scores[Players.HUMAN.value] for step in trajectory],
        Players.SHUTTER.value: [step.state.score_state.scores[Players.SHUTTER.value] for step in trajectory],
        #Players.NAO.value: [step.state.score_state.scores[Players.NAO.value] for step in trajectory]
        'NaoForHuman': [step.state.score_state.scores['NaoForHuman'] for step in trajectory],
        'NaoForShutter': [step.state.score_state.scores['NaoForShutter'] for step in trajectory],
        'Human+Nao': [step.state.score_state.scores[Players.HUMAN.value] + step.state.score_state.scores['NaoForHuman'] for step in trajectory],
        'Shutter+Nao': [step.state.score_state.scores[Players.SHUTTER.value] + step.state.score_state.scores['NaoForShutter'] for step in trajectory],
    }
    
    for player, scores in player_scores.items():
        ax.plot(range(len(scores)), scores, label=f'{player} Score')
    ax.set_title('Player Scores Over Trajectory')
    ax.set_xlabel('Trajectory Step Index')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    ax.legend()
    return ax

def _plot_welfare_values(
    ax: Axes, 
    trajectory: 'Trajectory', 
    human_expectations_value_function: 'ValueFunction', 
) -> Axes:
    if human_expectations_value_function is None:
        values_expected_from_model = None
        discount_factor = 1
    elif human_expectations_value_function.type == ValueFunctionType.MC_ROLLOUT:
        values_expected_from_model = None
        discount_factor = human_expectations_value_function.discount_factor
    elif human_expectations_value_function.type == ValueFunctionType.Q_FUNCTION:
        discount_factor = human_expectations_value_function.discount_factor
        values_expected_from_model = []
        for i, step in enumerate(trajectory):
            recent_states = trajectory.get_recent_states(index=i, num_recent_states=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
            values_expected_from_model.append(human_expectations_value_function(recent_states))
    else:
        raise ValueError(f"Unsupported ValueFunctionType: {human_expectations_value_function.type}")

    values_actual_from_traj = get_discounted_future_rewards_at_every_step(trajectory, discount_factor)

    if values_expected_from_model is not None:
        ax.plot(range(len(values_expected_from_model)), values_expected_from_model, label='Expected Values from Model')
    ax.plot(range(len(values_actual_from_traj)), values_actual_from_traj, label='Actual Values from Trajectory')
    ax.set_title('Expected vs Actual Values Over Trajectory')
    ax.set_xlabel('Trajectory Step Index')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    ax.legend()
    return ax

def plot_trajectory(
    trajectory: 'Trajectory', 
    human_expectations_value_function: 'ValueFunction', 
    output_dir: str, 
    filename: str
) -> str:
    
    plt.figure(figsize=(18, 24))

    # Subplot for player x positions (horizontal)
    ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=1, rowspan=1)
    ax1 = _plot_player_positions(ax1, trajectory, orientation='horizontal')

    # Subplot for actions over trajectory (all agents)
    ax2 = plt.subplot2grid((6, 2), (1, 0), colspan=1, rowspan=1, sharex=ax1)
    ax2 = _plot_player_actions(ax2, trajectory)

    # Subplot for NAO support
    ax6 = plt.subplot2grid((6, 2), (2, 0), colspan=1, rowspan=1, sharex=ax1)
    ax6 = _plot_nao_support(ax6, trajectory)

    # Subplot for events over trajectory
    ax3 = plt.subplot2grid((6, 2), (3, 0), colspan=1, rowspan=1, sharex=ax1)
    ax3 = _plot_events(ax3, trajectory)

    # Subplot for player scores over trajectory
    ax4 = plt.subplot2grid((6, 2), (4, 0), colspan=1, rowspan=1, sharex=ax1)
    ax4 = _plot_player_scores(ax4, trajectory)

    # Subplot for values expected from model and actual values from trajectory
    ax5 = plt.subplot2grid((6, 2), (5, 0), colspan=1, rowspan=1, sharex=ax1)
    ax5 = _plot_welfare_values(ax5, trajectory, human_expectations_value_function)

    # Subplot for player x positions (vertical)
    ax6 = plt.subplot2grid((6, 2), (0, 1), rowspan=6)
    ax6 = _plot_player_positions(ax6, trajectory, orientation='vertical')

    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()  # Close the figure to release memory
    return output_path



def smooth_trajectory_feature(trajectory_feature: List[float], window_size: int) -> List[float]:
    smoothed_trajectory_feature = []
    for i in range(len(trajectory_feature)):
        this_i_left = max(0, i-window_size) # don't go left of the beginning of the trajectory
        this_i_right = min(i+window_size+1, len(trajectory_feature)) # +1 to be inclusive, don't go right of the end of the trajectory
        windowed_feature = trajectory_feature[this_i_left:this_i_right]
        smooted_feature = sum(windowed_feature) / len(windowed_feature)
        smoothed_trajectory_feature.append(smooted_feature)
    return smoothed_trajectory_feature

def compute_welfare_reduction(actual_and_expected_values: List[List[float]]) -> List[float]:
    reduced_welfare_values = []
    for i in range(len(actual_and_expected_values)):
        step_idx, actual_welfare, expected_welfare = actual_and_expected_values[i]
        if expected_welfare is not None:
            reduced_welfare_values.append(actual_welfare - expected_welfare)
        else:
            reduced_welfare_values.append(actual_welfare)
    return reduced_welfare_values

def plot_welfare_values_and_diffs(values, title, output_dir, compare_smoothing=False):
    traj_step_indices = [val[0] for val in values]
    actual_values = [val[1] for val in values]
    expected_values = [val[2] for val in values]
    #diffs = [val[1] - val[2] for val in values]
    diffs = compute_welfare_reduction(values)

    if compare_smoothing:
        smooth_actual_values5 = smooth_trajectory_feature(actual_values, window_size=5)
        smooth_actual_values30 = smooth_trajectory_feature(actual_values, window_size=30)
        smooth_expected_values5 = smooth_trajectory_feature(expected_values, window_size=5)
        smooth_expected_values30 = smooth_trajectory_feature(expected_values, window_size=30)
        
        smooth_diffs5 = smooth_trajectory_feature(diffs, window_size=5)
        smooth_diffs30 = smooth_trajectory_feature(diffs, window_size=30)
        diff_of_smoothed5 = [smooth_actual_values5[i] - smooth_expected_values5[i] for i in range(len(smooth_actual_values5))]
        smooth_diff_of_smoothed5 = smooth_trajectory_feature(diff_of_smoothed5, window_size=5)
        diff_of_smoothed30 = [smooth_actual_values30[i] - smooth_expected_values30[i] for i in range(len(smooth_actual_values30))]
        smooth_diff_of_smoothed30 = smooth_trajectory_feature(diff_of_smoothed30, window_size=30)

        plt.figure(figsize=(24, 18))

        # Line plot for actual and expected values
        plt.subplot(3, 3, 1)
        plt.plot(traj_step_indices, actual_values, label='Actual Welfare', alpha=0.5)
        plt.plot(traj_step_indices, expected_values, label='Expected Welfare', alpha=0.5)
        plt.title(f"Expected versus Actual Welfare")
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Welfare')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for differences
        plt.subplot(3, 3, 2)
        plt.plot(traj_step_indices, diffs)
        plt.title(f'{title} - Differences (Actual - Expected)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Actual - Expected')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for actual vs expected with smoothing 5
        plt.subplot(3, 3, 3)
        plt.plot(traj_step_indices, smooth_actual_values5, label='Actual Welfare', alpha=0.5)
        plt.plot(traj_step_indices, smooth_expected_values5, label='Expected Welfare', alpha=0.5)
        plt.title(f"Expected versus Actual Welfare (smoothed over 5 frames)")
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Welfare')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()
        
        # Line plot for actual vs expected with smoothing 30
        plt.subplot(3, 3, 4)
        plt.plot(traj_step_indices, smooth_actual_values30, label='Actual Welfare', alpha=0.5)
        plt.plot(traj_step_indices, smooth_expected_values30, label='Expected Welfare', alpha=0.5)
        plt.title(f"Expected versus Actual Welfare (smoothed over 30 frames)")
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Welfare')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for smooth diffs 5
        plt.subplot(3, 3, 5)
        plt.plot(traj_step_indices, smooth_diffs5, label='Smooth Diffs 5', alpha=0.5)
        plt.title(f'{title} - Smooth Diffs (5)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Smooth Diffs 5')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for smooth diffs 30
        plt.subplot(3, 3, 6)
        plt.plot(traj_step_indices, smooth_diffs30, label='Smooth Diffs 30', alpha=0.5)
        plt.title(f'{title} - Smooth Diffs (30)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Smooth Diffs 30')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for smoothed diff of smoothed 5
        plt.subplot(3, 3, 7)
        plt.plot(traj_step_indices, smooth_diff_of_smoothed5, label='Smoothed Diff of Smoothed 5', alpha=0.5)
        plt.title(f'{title} - Smoothed Diff of Smoothed (5)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Smoothed Diff of Smoothed 5')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for smoothed diff of smoothed 30
        plt.subplot(3, 3, 8)
        plt.plot(traj_step_indices, smooth_diff_of_smoothed30, label='Smoothed Diff of Smoothed 30', alpha=0.5)
        plt.title(f'{title} - Smoothed Diff of Smoothed (30)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Smoothed Diff of Smoothed 30')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title}.png'))
        plt.close()  # Close the figure to release memory
    
    else:
        plt.figure(figsize=(24, 6))

        # Line plot for actual and expected values
        plt.subplot(1, 5, 1)
        plt.plot(traj_step_indices, actual_values, label='Actual', alpha=0.5)
        plt.plot(traj_step_indices, expected_values, label='Expected', alpha=0.5)
        plt.title(f'{title} - Actual vs Expected')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Line plot for differences
        plt.subplot(1, 5, 2)
        plt.plot(traj_step_indices, diffs)
        plt.title(f'{title} - Differences (Actual - Expected)')
        plt.xlabel('Trajectory Step Index')
        plt.ylabel('Actual - Expected')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Histogram for differences
        plt.subplot(1, 5, 3)
        plt.hist(diffs, bins=20)
        plt.title(f'{title} - Histogram')
        plt.xlabel('Actual - Expected')
        plt.ylabel('Frequency')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Histogram for actual values
        plt.subplot(1, 5, 4)
        plt.hist(actual_values, bins=20)
        plt.title(f'{title} - Actual Values Histogram')
        plt.xlabel('Actual Value')
        plt.ylabel('Frequency')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        # Histogram for expected values
        plt.subplot(1, 5, 5)
        plt.hist(expected_values, bins=20)
        plt.title(f'{title} - Expected Values Histogram')
        plt.xlabel('Expected Value')
        plt.ylabel('Frequency')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title}.png'))
        plt.close()  # Close the figure to release memory

def plot_all_welfare_reduction_methods(reward_values, future_value_values, past_rewards_values, past_rewards_and_future_value_values, output_dir, compare_smoothing=False):
    # Create plots for each method
    plot_welfare_values_and_diffs(reward_values, 'Reward Differences', output_dir, compare_smoothing)
    plot_welfare_values_and_diffs(future_value_values, 'Future Value Differences', output_dir, compare_smoothing)
    plot_welfare_values_and_diffs(past_rewards_values, 'Past Rewards Differences', output_dir, compare_smoothing)
    plot_welfare_values_and_diffs(past_rewards_and_future_value_values, 'Past Rewards and Future Value Differences', output_dir, compare_smoothing)

def plot_all_welfare_reduction_methods2(nonlocalized_values, single_step_values, past_values, output_dir, compare_smoothing=False):
    # Create plots for each method
    plot_welfare_values_and_diffs(nonlocalized_values, 'Nonlocalized Cumulative Reward Differences', output_dir, compare_smoothing)
    plot_welfare_values_and_diffs(single_step_values, 'Single Step Reward Differences', output_dir, compare_smoothing)
    plot_welfare_values_and_diffs(past_values, 'Complete Past Reward Differences', output_dir, compare_smoothing)

# Function to create a text-based histogram
def print_histogram(data, bins=10, title="Histogram"):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]
    max_count = max(hist)
    scale_factor = 50 / max_count  # Scale the histogram to fit within 50 characters

    print(title)
    for count, edge in zip(hist, bin_edges[:-1]):
        bar = '*' * int(count * scale_factor)
        print(f"{edge:>10.2f} - {edge + bin_width:>10.2f}: {bar}")

# Function to print a human-readable confusion matrix
from sklearn.metrics import confusion_matrix
def print_confusion_matrix(true: List[bool], predicted: List[bool]) -> np.ndarray:
    cm = confusion_matrix(true, predicted)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    return cm