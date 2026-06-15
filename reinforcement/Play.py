import argparse
import csv
import json
import os
import random
from datetime import datetime

import tensorflow as tf

from Configuration import Configuration
from Environment import Environment
from HttpClient import HttpClient
from CmdManager import CmdManager
from DdqnAgent import DoubleDeepQNetwork
from ExperimentConfig import ExperimentConfig


def get_attack_type():
    available_attacks = ["ICMP", "NESTEA", "SYN"]
    attack_type_index = random.randint(0, len(available_attacks) - 1)
    return available_attacks[attack_type_index]


def parse_hosts_arg(raw_hosts):
    if raw_hosts is None:
        return []
    stripped = raw_hosts.strip()
    if stripped == '' or stripped == '[]':
        return []
    tokens = stripped.lstrip('[').rstrip(']').split(',')
    parsed = [token.strip().strip("'\"") for token in tokens]
    return [host for host in parsed if host]


def load_pretrained_savedmodel_weights(ddqn_agent, model_path):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    saved = tf.saved_model.load(model_path)

    agent_weights = ddqn_agent.model.get_weights()
    n = len(agent_weights)
    extracted = [v.numpy() for v in saved.variables][:n]

    if len(extracted) != n:
        raise ValueError(
            f"SavedModel variable count is insufficient for injection: expected {n}, got {len(extracted)}"
        )

    for i, (agent_weight, loaded_weight) in enumerate(zip(agent_weights, extracted)):
        if agent_weight.shape != loaded_weight.shape:
            mismatch = f"Shape mismatch at variable {i}: {agent_weight.shape} vs {loaded_weight.shape}"
            print(f"(Play) ERROR: {mismatch}")
            raise AssertionError(mismatch)

    ddqn_agent.model.set_weights(extracted)
    ddqn_agent.model_target.set_weights(extracted)
    ddqn_agent.epsilon = 0.0
    ddqn_agent.epsilon_min = 0.0

    print("(Play) Pretrained weights loaded successfully")


def load_pretrained_weights(ddqn_agent, model_path):
    ddqn_agent.load_model(model_path)
    ddqn_agent.epsilon = 0.0
    ddqn_agent.epsilon_min = 0.0
    print("(Play) Checkpoint weights loaded successfully")


def load_pretrained_model(ddqn_agent, model_path):
    weights_file = model_path if model_path.endswith('.weights.h5') else f'{model_path}.weights.h5'
    if os.path.isfile(model_path) or os.path.isfile(weights_file):
        load_pretrained_weights(ddqn_agent, model_path)
        return
    if os.path.isdir(model_path):
        load_pretrained_savedmodel_weights(ddqn_agent, model_path)
        return
    raise FileNotFoundError(f"Model path not found: {model_path}")


def resolve_model_selection(args):
    model_path = args.model
    model_type = args.model_type
    if args.manifest:
        if not os.path.isfile(args.manifest):
            raise FileNotFoundError(f"Model manifest not found: {args.manifest}")
        with open(args.manifest) as manifest_file:
            manifest = json.load(manifest_file)
        model_type = model_type or manifest.get("model_type")
        if model_path is None:
            model_path = manifest.get("model_path")
            if model_path and not os.path.isabs(model_path):
                model_path = os.path.join(os.path.dirname(args.manifest), model_path)
    if model_path is None:
        raise ValueError("--model is required unless --manifest provides model_path")
    return model_type or "teacher", model_path


def build_sender_receiver_relation(env):
    relation = {}
    for host in env.normal_hosts:
        relation[host] = random.choice(env.servers)
    return relation


def build_attacker_victim_relation_and_types(env):
    attacker_victim_relation = {}
    attack_types = {}
    for attacker in env.attacker_hosts:
        attacker_victim_relation[attacker] = random.choice(env.victim_servers)
        attack_types[attacker] = get_attack_type()
    return attacker_victim_relation, attack_types


def save_play_results(config, model_path, attackers, episodes_payload, csv_rows):
    run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_path = f"{config.results_folder}/play_{run_time}.csv"
    json_path = f"{config.results_folder}/play_{run_time}.json"

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["episode", "step", "action_idx", "action_name", "reward"])
        writer.writeheader()
        writer.writerows(csv_rows)

    total_reward = 0.0
    total_steps = 0
    for episode in episodes_payload:
        total_reward += float(episode["total_reward"])
        total_steps += len(episode["steps"])
    overall_avg_reward = (total_reward / total_steps) if total_steps > 0 else 0.0

    payload = {
        "model_path": model_path,
        "attackers": attackers,
        "episodes": episodes_payload,
        "overall_avg_reward": overall_avg_reward
    }

    with open(json_path, 'w') as json_file:
        json.dump(payload, json_file, indent=2)

    print(f"(Play) CSV results saved: {csv_path}")
    print(f"(Play) JSON results saved: {json_path}")


def run_evaluation(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than 0")

    hosts_topo_file_name = "hosts-toplogy-6hosts.json"
    epsilon_decay = 0.999
    nbr_controlled_switches = 4

    model_type, model_path = resolve_model_selection(args)
    config = Configuration(hosts_topo_file_name, args.episodes, args.steps, epsilon_decay, nbr_controlled_switches)
    env = Environment(config)
    http_client = HttpClient(config)
    cmd = CmdManager(config)
    experiment_config = ExperimentConfig(
        {
            "experiment_name": "evaluation",
            "model_type": model_type,
            "distillation_method": "none",
        }
    )
    ddqn_agent = DoubleDeepQNetwork(
        config,
        env,
        http_client,
        is_controlled=False,
        is_prefilled_actions=False,
        experiment_config=experiment_config,
    )

    load_pretrained_model(ddqn_agent, model_path)

    pre_set_attackers = parse_hosts_arg(args.attackers)
    env.update_hosts()
    env.perform_setup(http_client, pre_set_attackers)
    ddqn_agent.set_actions(env.ACTIONS)

    print(f"(Play) Using attackers: {env.attacker_hosts}")

    csv_rows = []
    episodes_payload = []
    network_started = False

    try:
        # Startup order: network start -> wait_for_server -> get interfaces -> get_state
        cmd.start_network_in_background(
            env.servers,
            env.attacker_hosts,
            config.hosts_topo_file_name,
            config.nbr_controlled_switches
        )
        network_started = True

        print("(Play) Waiting for API server to accept connections...")
        if not http_client.wait_for_server(max_retries=60, initial_delay=2.0):
            raise RuntimeError("API server failed to start after maximum retries")

        env.update_hosts_ips(http_client)
        env.update_interfaces(http_client.get_switches_interfaces())
        tshark_interfaces_ids = env.get_tshark_interfaces_ids(cmd)

        for episode in range(1, args.episodes + 1):
            print(f"(Play) ========> Episode {episode} Started")

            sender_receiver_relation = build_sender_receiver_relation(env)
            attacker_victim_relation, attack_types = build_attacker_victim_relation_and_types(env)

            current_state = env.get_state(
                config,
                cmd,
                http_client,
                tshark_interfaces_ids,
                sender_receiver_relation,
                attacker_victim_relation,
                attack_types
            )

            episode_steps = []
            episode_total_reward = 0.0

            for step in range(1, args.steps + 1):
                action_idx, is_predicted = ddqn_agent.action(
                    step,
                    env.transform_state_dict_to_normalized_vector(current_state)
                )

                new_state, reward, done, _, _, _, _ = env.apply_action_controlled_switches(
                    config,
                    cmd,
                    http_client,
                    tshark_interfaces_ids,
                    sender_receiver_relation,
                    attacker_victim_relation,
                    attack_types,
                    action_idx,
                    is_predicted
                )

                reward_value = float(reward)
                action_name = env.ACTIONS[action_idx] if 0 <= action_idx < len(env.ACTIONS) else f"unknown:{action_idx}"

                csv_rows.append({
                    "episode": episode,
                    "step": step,
                    "action_idx": int(action_idx),
                    "action_name": action_name,
                    "reward": reward_value
                })

                episode_steps.append({
                    "step": step,
                    "action_idx": int(action_idx),
                    "action_name": action_name,
                    "reward": reward_value
                })

                episode_total_reward += reward_value
                current_state = new_state

                if done:
                    print(f"(Play) Episode {episode} ended early at step {step} (done=True)")
                    break

            step_count = len(episode_steps)
            avg_reward = (episode_total_reward / step_count) if step_count > 0 else 0.0
            episodes_payload.append({
                "episode": episode,
                "steps": episode_steps,
                "total_reward": episode_total_reward,
                "avg_reward": avg_reward
            })

            print(f"(Play) <======== Episode {episode} Ended (steps={step_count}, total_reward={episode_total_reward})")

    finally:
        cmd.stop_network()

    save_play_results(config, model_path, env.attacker_hosts, episodes_payload, csv_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained DDQN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="Path to .weights.h5 file (or base path without extension). Legacy SavedModel directories are also supported")
    parser.add_argument("--model-type", choices=["teacher", "student_a", "student_b"],
                        help="Architecture used by the checkpoint")
    parser.add_argument("--manifest", help="Model manifest providing model type and optionally model path")
    parser.add_argument("--attackers", default="['h1']", help="Attacker hosts list, e.g. ['h1']")
    parser.add_argument("--steps", type=int, default=5, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")

    run_evaluation(parser.parse_args())
