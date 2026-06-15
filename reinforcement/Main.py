import random
import numpy as np
from matplotlib.pyplot import cm
import csv
import hashlib
import json
import os
from decimal import Decimal
from Configuration import Configuration
from Environment import Environment
from HttpClient import HttpClient
from CmdManager import CmdManager
from DdqnAgent import DoubleDeepQNetwork
from ExperimentConfig import ExperimentConfig
import matplotlib.pyplot as plt
import shutil
import argparse
import tensorflow as tf
from tensorflow import keras


# Copies a CICFlowMeter output file to a new location.
# This is used to store the network flow metrics collected during an experiment step
def copy_cic_step_file(config, new_file_name):
    new_file_full_path = f"{config.cic_folder}/{new_file_name}"
    original_file_full_path = f"{config.cic_output_file_path}"
    try:
        shutil.copyfile(original_file_full_path, new_file_full_path)
    except FileNotFoundError:
        print(f"(Reinforcement) WARNING: CIC step file not found, skipping copy: {original_file_full_path}")


# Randomly selects an attack type from a predefined list of attacks.
def get_attack_type():
    # available_attacks = ["ICMP", "TCP", "UDP", "SYN"] # TDOO: Uncomment if random attack type
    available_attacks = ["ICMP"] # TODO: for testing purposes, currently just using ICMP attacks
    attack_type_index = random.randint(0, len(available_attacks) - 1)
    return available_attacks[attack_type_index]

def get_basic_metrics_headers():
    headers = ["tx_bytes",
               "rx_bytes",
               "bandwidth",
               "tx_packets",
               "rx_packets",
               "tx_packets_len",
               "rx_packets_len",
               "delivered_pkts",
               "loss_pct",
               "is_connected",
               "pkts_s",
               "bytes_s"]
    return headers

def get_network_metrics_headers():
    headers = ["avg_latency_s",
               "avg_packet_transmission_time_s",
               "throughput_bps",
               "avg_jitter_s"]
    return headers

SWITCHES_BW_HEADERS = None

def save_file_with_headers(filepath, data, headers, fmt='%.18e'):
    with open(filepath, 'w') as result_file:
        wr = csv.writer(result_file)
        wr.writerow(headers)
        np.savetxt(result_file, data, delimiter=',', fmt=fmt)


def safe_json_response(response, fallback):
    try:
        if hasattr(response, 'text') and response.text.strip():
            return response.json()
    except Exception:
        pass
    return fallback


def get_host_current_path(http_client, host):
    host_path_data = safe_json_response(http_client.get_host_path(host), {})
    return str(host_path_data.get('current', []))


def get_host_bw_value(http_client, host):
    host_bw_data = safe_json_response(http_client.get_host_bw(host), {})
    return Decimal(str(host_bw_data.get('bw', 0.0)))


def set_steps_xlim(steps):
    x_max = steps if steps > 1 else 2
    plt.xlim((1, x_max))


# Checks the network state data for anomalies such as NaN, infinite values, or negative metrics.
# If any issues are found, a warning file is generated to log the detected problems for further inspection.
def generate_warning_file_if_necessary(config, file_name, new_state):
    headers = get_basic_metrics_headers()
    headers.remove("bandwidth") # Cuz bandwidth is "Dec" type
    warnings = ""
    for host in new_state['host'].keys():
        for header in headers:
            if(np.isnan(new_state['host'][host][header])):
                warnings = warnings + f"\nISNAN: new_state['host'][{host}][{header}]={new_state['host'][host][header]}"
            elif (np.isinf(new_state['host'][host][header])):
                warnings = warnings + f"\nISINF: new_state['host'][{host}][{header}]={new_state['host'][host][header]}"
            elif (new_state['host'][host][header] < 0):
                warnings = warnings + f"\nNEGATIVE: new_state['host'][{host}][{header}]={new_state['host'][host][header]}"
    if len(warnings) > 0:
        warning_file = f"{config.current_train_folder}/{file_name}"
        f = open(warning_file, 'w')
        f.write(warnings)
        f.close()


def write_checkpoint_metadata(checkpoint_dir,
                              best_episode,
                              best_train_reward,
                              latest_episode,
                              checkpoint_every,
                              keep_last_checkpoints,
                              latest_file,
                              best_file):
    metadata = {
        "best_episode": best_episode,
        "best_train_reward": float(best_train_reward),
        "latest_episode": latest_episode,
        "checkpoint_every": checkpoint_every,
        "keep_last_checkpoints": keep_last_checkpoints,
        "latest_checkpoint": latest_file,
        "best_checkpoint": best_file,
    }
    metadata_path = f"{checkpoint_dir}/checkpoint_info.json"
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_manifest(config, experiment_config, env, ddqn_agent):
    teacher_path = experiment_config.teacher_weights
    teacher_sha256 = None
    if teacher_path:
        resolved_path = ddqn_agent._resolve_weights_file(teacher_path)
        if os.path.isfile(resolved_path):
            teacher_path = os.path.abspath(resolved_path)
            teacher_sha256 = sha256_file(resolved_path)

    manifest = {
        "experiment_name": experiment_config.experiment_name,
        "model_type": experiment_config.model_type,
        "distillation_method": experiment_config.distillation_method,
        "distillation_parameters": experiment_config.distillation,
        "teacher_checkpoint_path": teacher_path,
        "teacher_checkpoint_sha256": teacher_sha256,
        "seed": experiment_config.seed,
        "state_size": int(env.INPUT_SHAPE),
        "action_size": int(env.OUTPUT_SHAPE),
        "trainable_parameter_count": int(
            sum(np.prod(weight.shape) for weight in ddqn_agent.model.trainable_weights)
        ),
        "topology": config.hosts_topo_file_name,
        "controlled_switch_count": config.nbr_controlled_switches,
        "episodes": config.episodes,
        "steps": config.steps,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "model_path": "rl_model.weights.h5",
    }
    manifest_path = f"{config.rl_models_folder}/model_manifest.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


# The main block initializes the reinforcement learning environment and manages the experiment workflow.
# Key steps include:
# - Parsing command-line arguments for experiment configurations.
# - Initializing the network environment, RL agent, and other components.
# - Running multiple training episodes where the RL agent interacts with the environment.
# - Logging results and visualizing metrics such as packet loss, delay, and bandwidth usage
if __name__ == '__main__':

    # Parses command-line arguments to allow the user to customize the experiment setup.
    # Configurable parameters include the number of episodes, steps, and the epsilon decay rate.
    # Validation checks ensure the parameters are within acceptable ranges.
    parser = argparse.ArgumentParser(description="Main",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--attackers", help="Attacker hosts names. E.g: [h1]", required=False)
    parser.add_argument("-e", "--episodes", type=int, help="Number of episodes. E.g: 50", required=False)
    parser.add_argument("-s", "--steps", type=int, help="Number of steps. E.g: 100", required=False)
    parser.add_argument("-ed", "--epsilon-decay", type=float, help="Epsilon decay. E.g: 0.999", required=False)
    parser.add_argument("-ncs", "--nbr-controlled-switches", type=int, help="The number of controlled switches in the network", required=False)
    parser.add_argument("-c", "--controlled", action="store_true",
                        help="Whether to control action taking")
    parser.add_argument("-pfa", "--prefilled-actions", action="store_true",
                        help="Whether to use prefilled actions, from file 'prefilled-actions.txt'")
    parser.add_argument("-htf", "--hosts-topo-file",
                        help="When given, the provided JSON file in the 'input-data' folder will be used. E.g: hosts-topology-6hosts",
                        required=False)
    parser.add_argument("--checkpoint-every", type=int,
                        help="Save periodic checkpoints every N episodes")
    parser.add_argument("--keep-last-checkpoints", type=int,
                        help="How many periodic checkpoints to keep")
    parser.add_argument("--experiment-config", help="Path to an experiment JSON file")
    parser.add_argument("--model-type", choices=["teacher", "student_a", "student_b"])
    parser.add_argument("--distillation-method", choices=["none", "q_blend"])
    parser.add_argument("--teacher-weights", help="Path to teacher .weights.h5 checkpoint")
    parser.add_argument("--distillation-alpha", type=float, help="Q-blend Bellman target weight")
    parser.add_argument("--seed", type=int, help="Random seed")
    # Initializes the simulation environment and its components.
    # This involves setting up the network topology, controlled switches, and hosts.
    args = parser.parse_args()
    is_controlled = args.controlled
    is_prefilled_actions = args.prefilled_actions
    if is_controlled and is_prefilled_actions:
        raise Exception("Please use either '--controlled' flag or '--prefilled-actions' flag, but not both!")

    experiment_config = (
        ExperimentConfig.from_file(args.experiment_config, validate=False)
        if args.experiment_config
        else ExperimentConfig()
    )
    overrides = {
        "attackers": args.attackers,
        "episodes": args.episodes,
        "steps": args.steps,
        "epsilon_decay": args.epsilon_decay,
        "nbr_controlled_switches": args.nbr_controlled_switches,
        "hosts_topo_file": args.hosts_topo_file,
        "checkpoint_every": args.checkpoint_every,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "model_type": args.model_type,
        "distillation_method": args.distillation_method,
        "teacher_weights": args.teacher_weights,
        "seed": args.seed,
    }
    experiment_values = experiment_config.to_dict()
    for key, value in overrides.items():
        if value is not None:
            experiment_values[key] = value
    if args.distillation_alpha is not None:
        experiment_values["distillation"] = {
            **experiment_values.get("distillation", {}),
            "alpha": args.distillation_alpha,
        }
    experiment_config = ExperimentConfig(
        experiment_values, source_path=experiment_config.source_path
    )

    if experiment_config.seed is not None:
        random.seed(experiment_config.seed)
        np.random.seed(experiment_config.seed)
        tf.keras.utils.set_random_seed(experiment_config.seed)

    pre_set_attackers = []
    if experiment_config.attackers not in ("", "[]"):
        raw_attackers = experiment_config.attackers.lstrip("[").rstrip("]").split(',')
        pre_set_attackers = [a.strip().strip("'\"") for a in raw_attackers]
    if is_controlled:
        print('(Reinforcement) ================> Main Started with "controlled actions"')
    elif is_prefilled_actions:
        print('(Reinforcement) ================> Main Started with "prefilled actions"')
    else:
        print('(Reinforcement) ================> Main Started')
    hosts_topo_file_name = experiment_config.hosts_topo_file
    if not hosts_topo_file_name.lower().endswith(".json"):
        hosts_topo_file_name += ".json"
    episodes = experiment_config.episodes
    steps = experiment_config.steps
    epsilon_decay = float(experiment_config.epsilon_decay)
    nbr_controlled_switches = experiment_config.nbr_controlled_switches
    checkpoint_every = experiment_config.checkpoint_every
    keep_last_checkpoints = experiment_config.keep_last_checkpoints
    print(f'(Reinforcement) ==================> Experiment: {experiment_config.experiment_name}')
    print(f'(Reinforcement) ==================> Model type: {experiment_config.model_type}')
    print(f'(Reinforcement) ==================> Distillation: {experiment_config.distillation_method}')
    print(f'(Reinforcement) ==================> Episodes: {episodes}')
    print(f'(Reinforcement) ==================> Steps: {steps}')
    print(f'(Reinforcement) ==================> Epsilon decay: {epsilon_decay}')
    print(f'(Reinforcement) ==================> Number of controlled switches: {nbr_controlled_switches}')
    print(f'(Reinforcement) ==================> Checkpoint every: {checkpoint_every} episode(s)')
    print(f'(Reinforcement) ==================> Keep periodic checkpoints: {keep_last_checkpoints}')

    config = Configuration(hosts_topo_file_name, episodes, steps, epsilon_decay, nbr_controlled_switches)
    experiment_config.write(f"{config.configs_folder}/experiment.json")
    env = Environment(config)
    cmd = CmdManager(config)
    http_client = HttpClient(config)
    tot_rewards = 0
    total_rewards_per_episode = []
    epsilons = []
    ddqn_agent = DoubleDeepQNetwork(
        config,
        env,
        http_client,
        is_controlled,
        is_prefilled_actions,
        experiment_config=experiment_config,
    )
    write_model_manifest(config, experiment_config, env, ddqn_agent)
    checkpoint_dir = f"{config.rl_models_folder}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    periodic_checkpoint_files = []
    best_train_reward = float('-inf')
    best_episode = 0
    latest_checkpoint_file = f"{checkpoint_dir}/latest.weights.h5"
    best_checkpoint_file = f"{checkpoint_dir}/best.weights.h5"

    global_vars_to_print = {
        "max_attacker": {},
        "max_host": {},
        "max_server": {},
    }
    for header in get_basic_metrics_headers():
        global_vars_to_print["max_attacker"][header] = 0
        global_vars_to_print["max_host"][header] = 0
        global_vars_to_print["max_server"][header] = 0

    for header in get_network_metrics_headers():
        global_vars_to_print["max_host"][header] = 0

    # Iterates through the specified number of episodes for training the reinforcement learning agent.
    # Each episode involves multiple interaction steps where the agent takes actions and receives feedback.
    # Traffic and performance metrics are collected and saved at the end of each episode.
    for episode in range(1, env.episodes + 1):
        tot_rewards = 0
        episode_index = episode - 1
        current_state = env.reset()

        episode_rewards = []
        ddqn_agent.episode_loss = []
        ddqn_agent.episode_loss = []
        episode_avg_packet_loss = []
        episode_avg_real_delays = []
        episode_avg_latencys = []
        episode_avg_jitters = []

        print(f'(Reinforcement) ==================> Episode {episode} Started')

        env.update_hosts()

        env.perform_setup(http_client, pre_set_attackers)

        ddqn_agent.set_actions(env.ACTIONS)

        cmd.start_network_in_background(env.servers, env.attacker_hosts, config.hosts_topo_file_name, nbr_controlled_switches)

        print("(Reinforcement) Waiting for API server to accept connections...")
        if not http_client.wait_for_server(max_retries=60, initial_delay=2.0):
            raise Exception("API server failed to start after maximum retries")

        env.update_hosts_ips(http_client)

        env.update_interfaces(http_client.get_switches_interfaces())

        tshark_interfaces_ids = env.get_tshark_interfaces_ids(cmd)

        sender_receiver_relation = {}
        for host in env.normal_hosts:
            server_index = random.randint(0, len(env.servers) - 1)
            server = env.servers[server_index]
            sender_receiver_relation[host] = server

        attacker_victim_relation = {}
        attack_types = {}
        for attacker in env.attacker_hosts:
            victim_server_index = random.randint(0, len(env.victim_servers) - 1)
            victim_server = env.victim_servers[victim_server_index]
            attacker_victim_relation[attacker] = victim_server
            attack_types[attacker] = get_attack_type()

        # variables for each host

        attacker_state_variables = {}
        for attacker in env.attacker_hosts:
            cols = env.NBR_HOST_STATE_METRICS + 1
            attacker_state_variables[attacker] = {
                'filename': f'attacker_{attacker}_attackType_{attack_types[attacker]}.csv',
                'data': np.empty((env.steps, cols), dtype=object)
            }
            attacker_state_variables[attacker]['data'][:, 0:(cols - 1)] = 0.0
            attacker_state_variables[attacker]['data'][:, (cols - 1)] = ""
        server_state_variables = {}
        for server in env.servers:
            attacker_suffix = ""
            for attacker in env.attacker_hosts:
                if attacker_victim_relation[attacker] == server:
                    attacker_suffix = f"{attacker_suffix}_attacker_{attacker}_type_{attack_types[attacker]}"
            server_state_variables[server] = {
                'filename': f'server_{server}{attacker_suffix}.csv',
                'data': np.zeros((env.steps, env.NBR_HOST_STATE_METRICS))
            }
        normal_host_state_variables = {}
        for host in env.normal_hosts:
            cols = env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics + 1
            normal_host_state_variables[host] = {
                'filename': f'host_{host}.csv',
                'data': np.empty((env.steps, cols), dtype=object)
            }
            normal_host_state_variables[host]['data'][:, 0:(cols - 1)] = 0.0
            normal_host_state_variables[host]['data'][:, (cols - 1)] = ""

        switches_bw_variables = {
            'filename': f'switches_bw.csv',
            'data': np.zeros((env.steps, env.nbr_routing_switches + (env.nbr_controlled_switches * env.nbr_controlled_switches)))
        }

        episode_hosts_bw = {}
        for host in env.hosts:
            episode_hosts_bw[host] = {'data': []}

        print(f'(Reinforcement) ====================> Init Step Started')

        new_state = env.get_state(config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation,
                                  attacker_victim_relation, attack_types)
        current_state = new_state
        env.last_recorded_delay = env.calculate_delay(current_state)
        env.last_recorded_latency = env.calculate_latency(current_state)
        env.last_recorded_jitter = env.calculate_jitter(current_state)
        env.before_last_recorded_delay = env.last_recorded_delay

        for i in range(1, 1):
            print(f'(Reinforcement) ====================> Init Step Started - Additional {i}')
            new_state = env.get_state(config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation,
                                      attacker_victim_relation, attack_types)
            current_state = new_state
            print(f'(Reinforcement) <==================== Init Step Ended - Additional {i}')
        print(current_state)

        print(f'(Reinforcement) <==================== Init Step Ended')

        # Executes a series of steps within each episode.
        # During each step, the RL agent selects an action, and the environment updates its state accordingly.
        for step in range(1, env.steps + 1):

            # The RL agent selects an action either based on its policy or by exploration (random actions).
            # The selected action is applied to the environment, which responds with a new state and reward.
            # The action's effectiveness is evaluated based on the resulting network performance metrics.
            print(f'(Reinforcement) ====================> Step {step} (of episode {episode}) Started')

            action, is_predicted = ddqn_agent.action(step, env.transform_state_dict_to_normalized_vector(current_state))

            new_state, reward, done, avg_packet_loss, avg_real_delays, avg_latency, avg_jitter = env.apply_action_controlled_switches(
                config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation, attacker_victim_relation,
                attack_types, action, is_predicted)

            episode_avg_packet_loss.append(avg_packet_loss)
            episode_avg_real_delays.append(avg_real_delays)
            episode_avg_latencys.append(avg_latency)
            episode_avg_jitters.append(avg_jitter)
            print(new_state)

            generate_warning_file_if_necessary(config, f"Episode {episode} - Step {step} - Warning.txt", new_state)

            tot_rewards += reward
            episode_rewards.append(reward)

            ddqn_agent.store(env.transform_state_dict_to_normalized_vector(current_state), action,
                             reward, env.transform_state_dict_to_normalized_vector(new_state), done)

            current_state = new_state

            # Experience Replay
            if len(ddqn_agent.memory) > ddqn_agent.batch_size:
                ddqn_agent.experience_replay(ddqn_agent.batch_size, episode=episode, step=step)
            else:
                ddqn_agent.episode_loss.append(1)

            if done or (step % ddqn_agent.update_target_each == 0):
                ddqn_agent.update_target_from_model()

            do_break = False
            if done or step == env.steps:
                total_rewards_per_episode.append(tot_rewards)
                epsilons.append(ddqn_agent.epsilon)
                do_break = True

            step_index = step - 1

            #############################################################################################################
            # filling state information of each host in each step in order to be saved in a csv file after each episode #
            #############################################################################################################
            for attacker in env.attacker_hosts:
                attacker_data = new_state['host'].get(attacker)
                if attacker_data is None:
                    print(f"(Reinforcement) WARNING: Missing state for attacker host {attacker} at step {step}")
                    continue
                arr = np.zeros(env.NBR_HOST_STATE_METRICS)
                i = 0
                for header in get_basic_metrics_headers():
                    arr[i] = attacker_data.get(header, 0.0)
                    i = i + 1
                attacker_state_variables[attacker]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
                ####################new_state['host']#####################
                for header in get_basic_metrics_headers():
                    global_vars_to_print['max_attacker'][header] = max(
                        global_vars_to_print['max_attacker'][header], attacker_data.get(header, 0.0))
                attacker_state_variables[attacker]['data'][step_index, env.NBR_HOST_STATE_METRICS] = get_host_current_path(http_client, attacker)
            for server in env.servers:
                server_data = new_state['host'].get(server)
                if server_data is None:
                    print(f"(Reinforcement) WARNING: Missing state for server host {server} at step {step}")
                    continue
                arr = np.zeros(env.NBR_HOST_STATE_METRICS)
                i = 0
                for header in get_basic_metrics_headers():
                    arr[i] = server_data.get(header, 0.0)
                    i = i + 1
                server_state_variables[server]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
                ####################new_state['host']#####################
                for header in get_basic_metrics_headers():
                    global_vars_to_print['max_server'][header] = max(
                        global_vars_to_print['max_server'][header], server_data.get(header, 0.0))

            ######################################## normal host state variable##############################
            for normal_host in env.normal_hosts:
                normal_host_data = new_state['host'].get(normal_host)
                if normal_host_data is None:
                    print(f"(Reinforcement) WARNING: Missing state for normal host {normal_host} at step {step}")
                    continue
                arr = np.zeros(env.NBR_HOST_STATE_METRICS)
                i = 0
                for header in get_basic_metrics_headers():
                    arr[i] = normal_host_data.get(header, 0.0)
                    i = i + 1
                normal_host_state_variables[normal_host]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
                normal_host_network_metrics = normal_host_data.get('non_server_data', {}).get('network_metrics', {})
                arr = np.zeros(env.nbr_of_network_metrics)
                i = 0
                for header in get_network_metrics_headers():
                    arr[i] = normal_host_network_metrics.get(header, 0.0)
                    i = i + 1
                normal_host_state_variables[normal_host]['data'][step_index, env.NBR_HOST_STATE_METRICS:(env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics)] = arr
                normal_host_state_variables[normal_host]['data'][step_index, env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics] = get_host_current_path(http_client, normal_host)
                ####################new_state['host']#####################
                for header in get_basic_metrics_headers():
                    global_vars_to_print['max_host'][header] = max(
                        global_vars_to_print['max_host'][header], normal_host_data.get(header, 0.0))
                for header in get_network_metrics_headers():
                    global_vars_to_print['max_host'][header] = max(
                        global_vars_to_print['max_host'][header], normal_host_network_metrics.get(header, 0.0))
            ######################################## Switches BW variables ##############################
            if SWITCHES_BW_HEADERS is None:
                SWITCHES_BW_HEADERS = []
                for src_switch in new_state['routing'].keys():
                    for dst_switch in new_state['routing'][src_switch].keys():
                        SWITCHES_BW_HEADERS.append(f"{src_switch} -> {dst_switch}")
                for src_switch in new_state['controlled'].keys():
                    for dst_switch in new_state['controlled'][src_switch].keys():
                        SWITCHES_BW_HEADERS.append(f"{src_switch} -> {dst_switch}")
            arr = np.zeros(env.nbr_routing_switches + (env.nbr_controlled_switches * env.nbr_controlled_switches))
            i = 0
            for src_switch in new_state['routing'].keys():
                for dst_switch in new_state['routing'][src_switch].keys():
                    arr[i] = new_state['routing'][src_switch][dst_switch]['bw']
                    i = i + 1
            for src_switch in new_state['controlled'].keys():
                for dst_switch in new_state['controlled'][src_switch].keys():
                    arr[i] = new_state['controlled'][src_switch][dst_switch]['bw']
                    i = i + 1
            switches_bw_variables['data'][step_index, :] = arr

            for host in env.hosts:
                episode_hosts_bw[host]['data'].append(get_host_bw_value(http_client, host))

            copy_cic_step_file(config, f"Episode {episode} - Step {step} - CIC results.csv")

            print(f'(Reinforcement) <==================== Step {step} (of episode {episode}) Ended')

            if do_break:
                break

        for normal_host in env.normal_hosts:
            headers = get_basic_metrics_headers() + get_network_metrics_headers() + ["current_path"]
            save_file_with_headers(f"{config.data_folder}/Episode {episode} - {normal_host_state_variables[normal_host]['filename']}", normal_host_state_variables[normal_host]['data'], headers, fmt='%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%s')
        for server in env.servers:
            save_file_with_headers(f"{config.data_folder}/Episode {episode} - {server_state_variables[server]['filename']}", server_state_variables[server]['data'], get_basic_metrics_headers())
        for attacker in env.attacker_hosts:
            headers = get_basic_metrics_headers() + ["current_path"]
            save_file_with_headers(f"{config.data_folder}/Episode {episode} - {attacker_state_variables[attacker]['filename']}", attacker_state_variables[attacker]['data'], headers, fmt='%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%s')
        save_file_with_headers(f"{config.data_folder}/Episode {episode} - {switches_bw_variables['filename']}", switches_bw_variables['data'], SWITCHES_BW_HEADERS)
        save_file_with_headers(f"{config.data_folder}/Episode {episode} - Actions", env.episode_actions_text_list, ["Action", "Message"], fmt='%s')
        # Generates plots to visualize the performance of the RL agent across multiple episodes.
        # Graphs include total reward per episode, packet loss, delay, and bandwidth usage.
        fig1 = plt.figure(f"Episode {episode} Reward")
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, color='b', label='rewards')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title(f"Episode {episode} Reward")
        fig1.savefig(f"{config.figures_folder}/Episode {episode} - Reward.png")
        plt.close(fig1)

        fig1 = plt.figure(f"Episode {episode} Loss Function")
        plt.plot(range(1, len(ddqn_agent.episode_loss) + 1), ddqn_agent.episode_loss, color='r', label='loss function')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("Loss function")
        plt.title(f"Episode {episode} Loss Function")
        fig1.savefig(f"{config.figures_folder}/Episode {episode} - Loss Function.png")
        plt.close(fig1)

        fig3_1 = plt.figure(f"Episode {episode} PKT loss")
        plt.plot(range(1, len(episode_avg_packet_loss) + 1), [100 * x for x in episode_avg_packet_loss], color='b', label='pkt loss')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("PKT loss")
        plt.title(f"Episode {episode} PKT loss")
        fig3_1.savefig(f"{config.figures_folder}/Episode {episode} - PKT loss.png")
        plt.close(fig3_1)

        fig3_2 = plt.figure(f"Episode {episode} AVG delay")
        plt.plot(range(1, len(episode_avg_real_delays) + 1), episode_avg_real_delays, color='r', label='avg delay')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("AVG delay")
        plt.title(f"Episode {episode} AVG delay")
        fig3_2.savefig(f"{config.figures_folder}/Episode {episode} - AVG delay.png")
        plt.close(fig3_2)

        fig3_3 = plt.figure(f"Episode {episode} AVG latency")
        plt.plot(range(1, len(episode_avg_latencys) + 1), episode_avg_latencys, color='g', label='avg latency')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("AVG latency")
        plt.title(f"Episode {episode} AVG latency")
        fig3_3.savefig(f"{config.figures_folder}/Episode {episode} - AVG latency.png")
        plt.close(fig3_3)

        fig3_4 = plt.figure(f"Episode {episode} AVG jitter")
        plt.plot(range(1, len(episode_avg_jitters) + 1), episode_avg_jitters, color='m', label='avg jitter')
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("AVG jitter")
        plt.title(f"Episode {episode} AVG jitter")
        fig3_4.savefig(f"{config.figures_folder}/Episode {episode} - AVG jitter.png")
        plt.close(fig3_4)

        fig5 = plt.figure(f"Episode {episode} Hosts BW")
        for host in env.hosts:
            host_label = f'{host}'
            if host in env.servers:
                host_label = f'{host_label} (server)'
            elif host in env.attacker_hosts:
                host_label = f'{host_label} (attacker {attack_types[host]})'
            plt.plot(range(1, len(episode_hosts_bw[host]['data']) + 1), episode_hosts_bw[host]['data'], label=host_label)
        plt.legend()
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("BW")
        plt.title(f"Episode {episode} Hosts BW")
        fig5.savefig(f"{config.figures_folder}/Episode {episode} - Hosts BW")
        plt.close(fig5)

        fig6 = plt.figure(f"Episode {episode} Switches BW")
        color = iter(cm.rainbow(np.linspace(0, 1, len(SWITCHES_BW_HEADERS))))
        for i in range(len(SWITCHES_BW_HEADERS)):
            switch_label = SWITCHES_BW_HEADERS[i]
            c = next(color)
            plt.plot(range(1, len(switches_bw_variables['data'][:,i]) + 1), switches_bw_variables['data'][:,i],
                     label=switch_label, c=c)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        set_steps_xlim(env.steps)
        plt.xlabel("Steps")
        plt.ylabel("BW")
        plt.title(f"Episode {episode} Switches BW")
        fig6.savefig(f"{config.figures_folder}/Episode {episode} - Switches BW", bbox_inches='tight')
        plt.close(fig6)

        print(f'(Reinforcement) <================== Episode {episode} Ended')

        ddqn_agent.save_model(f"{checkpoint_dir}/latest")

        if (episode % checkpoint_every) == 0:
            periodic_checkpoint_base = f"{checkpoint_dir}/ep_{episode:04d}"
            ddqn_agent.save_model(periodic_checkpoint_base)
            periodic_checkpoint_files.append(f"{periodic_checkpoint_base}.weights.h5")

            while len(periodic_checkpoint_files) > keep_last_checkpoints:
                oldest = periodic_checkpoint_files.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)

        if float(tot_rewards) > best_train_reward:
            best_train_reward = float(tot_rewards)
            best_episode = episode
            ddqn_agent.save_model(f"{checkpoint_dir}/best")
            print(f"(Reinforcement) ==> New best training checkpoint at episode {episode} (reward={best_train_reward})")

        write_checkpoint_metadata(
            checkpoint_dir,
            best_episode,
            best_train_reward,
            episode,
            checkpoint_every,
            keep_last_checkpoints,
            latest_checkpoint_file,
            best_checkpoint_file,
        )

        plt.close('all')
        cmd.stop_network()


    ddqn_agent.save_model(f"{config.rl_models_folder}/rl_model")

    fig = plt.figure(f"Results per Episode")
    plt.plot(range(1, env.episodes + 1), total_rewards_per_episode, color='blue', label='Total rewards per episode')
    plt.axhline(y=max(total_rewards_per_episode), color='r', linestyle='-', label='Max total reward')
    eps_graph = [max(total_rewards_per_episode) * x for x in epsilons]
    plt.plot(range(1, env.episodes + 1), eps_graph, color='g', linestyle='-', label='Epsilon')
    plt.legend()
    plt.xlabel("Episode")
    set_steps_xlim(env.episodes)
    plt.ylim((min(total_rewards_per_episode), 1.1 * max(total_rewards_per_episode)))
    plt.title(f"Results per Episode")
    fig.savefig(f"{config.figures_folder}/Last - total rewards and epsilon.png")
    plt.close('all')

    print(global_vars_to_print)

    print('(Reinforcement) ================> Main Ended')
