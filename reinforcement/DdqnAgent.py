import csv
import numpy as np
import random
import os
from Util import Util
from ModelFactory import build_model
from distillation import build_strategy

from collections import deque


class DoubleDeepQNetwork():

    # Initializes the Double Deep Q-Network (DDQN) agent with the specified configuration and environment.
    # Sets up hyperparameters like epsilon for exploration, gamma for discounting, and batch size for training.
    # Initializes the neural network models for both the primary and target Q-networks.
    # If prefilled actions are provided, they are loaded for controlled execution.
    def __init__(
        self,
        config,
        env,
        http_client,
        is_controlled,
        is_prefilled_actions,
        experiment_config=None,
    ):
        self.ACTIONS = None
        self.config = config
        self.env = env
        self.http_client = http_client
        self.is_controlled = is_controlled
        self.is_prefilled_actions = is_prefilled_actions
        self.nS = self.env.INPUT_SHAPE
        self.nA = self.env.OUTPUT_SHAPE
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # Experimentation Epsilon decay Values:
        #  50 episode X  50 step ==> 2500 ==> 0.998
        #  50 episode X 100 step ==> 5000 ==> 0.999
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = 0.01
        # self.tau = 0.125 # TODO: Possible future improvement
        self.batch_size = 8
        self.memory_size = 2500
        self.memory = deque(maxlen=self.memory_size)
        self.update_target_each = 10 # steps
        self.model_type = (
            experiment_config.model_type if experiment_config is not None else "teacher"
        )
        self.distillation_method = (
            experiment_config.distillation_method if experiment_config is not None else "none"
        )
        distillation_parameters = (
            experiment_config.distillation if experiment_config is not None else {}
        )
        self.distillation_strategy = build_strategy(
            self.distillation_method, distillation_parameters
        )
        self.teacher_weights = (
            experiment_config.teacher_weights if experiment_config is not None else None
        )
        self.replay_updates = 0
        self.distillation_metrics_file = None

        # model is updated instantly
        # target_model updated after each batch
        self.model = self.build_model()
        self.model_target = self.build_model()
        self.update_target_from_model()  # Update weights
        self.teacher_model = self._build_teacher_model_if_required()
        self.loss = []
        self.episode_loss = []

        if is_prefilled_actions:
            self.prefilled_actions = self.read_lines_from_file(config.prefilled_actions_file)
            print("<------> Actions are prefilled:")
            for prefilled_action in self.prefilled_actions:
                print(f"---------> {prefilled_action}")

    @DeprecationWarning
    # Reads a list of actions from a specified file.
    # Each action is expected to be defined on a separate line, and empty lines are ignored.
    def read_lines_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
            return lines

    def build_model(self):
        return build_model(self.model_type, self.nS, self.nA, self.learning_rate)

    def _build_teacher_model_if_required(self):
        if not self.distillation_strategy.requires_teacher:
            return None
        if not self.teacher_weights:
            raise ValueError(
                f"Distillation method '{self.distillation_method}' requires teacher weights"
            )
        teacher_model = build_model("teacher", self.nS, self.nA, self.learning_rate)
        self._load_weights_checked(teacher_model, self.teacher_weights, "teacher")
        teacher_model.trainable = False
        for layer in teacher_model.layers:
            layer.trainable = False
        return teacher_model

    def _load_weights_checked(self, model, filename, label):
        weights_file = self._resolve_weights_file(filename)
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"{label.capitalize()} weights file not found: {weights_file}")
        try:
            model.load_weights(weights_file)
        except Exception as exc:
            raise ValueError(
                f"Incompatible {label} checkpoint '{weights_file}' for "
                f"model_type='{model.name}', state_size={self.nS}, action_size={self.nA}: {exc}"
            ) from exc
        return weights_file

    @staticmethod
    def _resolve_weights_file(filename):
        if filename.endswith(".weights.h5") or os.path.isfile(filename):
            return filename
        extensionless_weights = f"{filename}.weights.h5"
        return extensionless_weights if os.path.exists(extensionless_weights) else filename

    # Copies the weights from the primary model to the target model.
    # The target model is used for more stable training updates.
    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())
        print(f'<------> Target Model Updated')

    # Selects an action based on the current state and the exploration-exploitation trade-off.
    # If a random number is below epsilon, the agent explores a random action.
    # Otherwise, it uses the model to predict the best action.
    # Returns the selected action and a flag indicating if the action was predicted.
    def action(self, step, state):
        print('<--------------------------------------------------------->\n\n')
        if self.is_controlled:
            return self.do_controlled_prompt()
        if self.is_prefilled_actions:
            return self.do_action_from_prefilled(step)
        # return self.nA - 1 # TODO: Testing purposes only, always take no action
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.nA)  # Explore
            print(f'<------> Taking action randomly: {action}')
            return action, False
        action_vals = self.model.predict(np.reshape(state, [1, self.nS]), verbose=0)
        # Exploit: Use the NN to predict the correct action from this state
        #   action_vals [0] ==> because the output shape is (1, 9) meaning one line and 12 column
        #   so to get the probabilities of taking each of the 9 actions we use the first line (index 0)
        action = np.argmax(action_vals[0])
        print(f'<------> Taking action with predict: {action}')
        return action, True

    @DeprecationWarning
    def do_controlled_prompt(self):
        action = -1
        while self.is_controlled and action < 0:
            print("Avilable actions:")
            for i in range(len(self.ACTIONS)):
                ACTION = self.ACTIONS[i]
                ACTIONS_splitted = ACTION.split(':')
                if ACTIONS_splitted[0] == "redirect":
                    current_switches, dst_switch = self.get_controlled_redirect_action_with_dist(ACTIONS_splitted)
                    print(f" - {i}: {ACTION} (redirect from {current_switches} to {dst_switch})")
                else:
                    print(f" - {i}: {ACTION}")
            action = int(input("Enter action index:"))
            if 0 <= action < self.nA:
                return action, True
            print(f'<------> Action ({action}) is not recognised, please try again!')
            action = -1

    @DeprecationWarning
    # Executes a predefined action from a list of prefilled actions loaded from a file.
    # If no valid action is provided for the current step, a default "Do Nothing" action is taken.
    def do_action_from_prefilled(self, step):
        custom_action = self.get_step_index_action_or_nothing(step)
        action = -1
        if custom_action == Util.nothing_action():
            action = len(self.ACTIONS) - 1
        else:
            if custom_action.startswith("bw"):
                action = self.ACTIONS.index(custom_action)
            elif custom_action.startswith("redirect"):
                custom_action_splitted = custom_action.split(':')
                host_name = custom_action_splitted[1]
                dst_switch = custom_action_splitted[3]
                custom_action_parsed = Util.redirect_action(host_name, dst_switch)
                print(f'---------> {custom_action} corresponds to {custom_action_parsed}')
                action = self.ACTIONS.index(custom_action_parsed)
        if action == -1:
            action = len(self.ACTIONS) - 1
        print(f'<------> Taking action with prefilled: {action}')
        return action, True

    def get_step_index_action_or_nothing(self, step):
        step_index = step - 1
        custom_action = Util.nothing_action()
        if step_index < len(self.prefilled_actions):
            custom_action = self.prefilled_actions[step_index]
        return custom_action

    # Parses a redirect action to extract the host and destination switch involved.
    # Returns the host name and destination switch as a tuple for further processing.
    def get_controlled_redirect_action_with_dist(self, ACTIONS_splitted):
        host_name = ACTIONS_splitted[1]
        dst_switch = ACTIONS_splitted[3]
        return (host_name, dst_switch)

    # For testing purposes:
    # Selects the best action from the model predictions without exploration.
    # Used during testing or evaluation to purely exploit the learned policy.
    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(np.reshape(state, [1, self.nS]), verbose=0)
        return np.argmax(action_vals[0])

    # Stores an experience tuple (state, action, reward, next state, done) into the replay memory.
    # The memory is used for experience replay during training.
    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    # Trains the model using a random batch of experiences from the replay memory.
    # Each experience is processed to compute the target Q-value using the Bellman equation.
    # The model is trained using the generated batch, and epsilon is decayed after each iteration.
    # Double DQN is implemented here by using the target model for stable Q-value estimation.
    def experience_replay(self, batch_size, episode=None, step=None):
        # Execute the experience replay
        # each element of memory is cur_state, action, reward, new_state, done(after each step)
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        minibatch_array = np.array(minibatch, dtype=object)
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(minibatch_array)):  # Creating the state and next state np arrays
            st = np.append(st, minibatch_array[i, 0].reshape(1, self.nS), axis=0)
            nst = np.append(nst, minibatch_array[i, 3].reshape(1, self.nS), axis=0)
        st_predict = self.model.predict(st, verbose=0)
        nst_predict = self.model.predict(nst, verbose=0)
        nst_predict_target = self.model_target.predict(nst, verbose=0)
        for state, _, _, _, _ in minibatch:
            x.append(state)
        targets, metrics = self.distillation_strategy.construct_targets(
            minibatch=minibatch,
            student_q_values=st_predict,
            next_student_q_values=nst_predict,
            next_target_q_values=nst_predict_target,
            gamma=self.gamma,
            teacher_model=self.teacher_model,
            states=st,
        )
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.asarray(targets)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        self.replay_updates += 1
        if metrics is not None:
            self._record_distillation_metrics(metrics, episode, step)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
            self.episode_loss.append(hist.history['loss'][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("<------> New Epsilon value: " + str(self.epsilon))

        return metrics

    def _record_distillation_metrics(self, metrics, episode, step):
        if self.distillation_metrics_file is None:
            self.distillation_metrics_file = os.path.join(
                self.config.rl_stats_folder, "distillation_metrics.csv"
            )
        file_exists = os.path.exists(self.distillation_metrics_file)
        fieldnames = [
            "episode",
            "step",
            "replay_update_number",
            "student_teacher_q_mse",
            "mean_abs_student_q",
            "max_abs_student_q",
        ]
        row = {
            "episode": "" if episode is None else episode,
            "step": "" if step is None else step,
            "replay_update_number": self.replay_updates,
            **metrics,
        }
        with open(self.distillation_metrics_file, "a", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_model(self, filename):
        weights_file = filename if filename.endswith('.weights.h5') else f'{filename}.weights.h5'
        self.model.save_weights(weights_file)

    def load_model(self, filename):
        self.model = self.build_model()
        self._load_weights_checked(self.model, filename, self.model_type)
        self.model_target = self.build_model()
        self.model_target.set_weights(self.model.get_weights())

    def set_actions(self, ACTIONS):
        self.ACTIONS = ACTIONS
