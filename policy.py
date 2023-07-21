from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

from gymnasium import spaces
import torch as th
from torch import nn
import numpy as np
from functools import partial


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback
from mpc import DistributionalModelPredictiveControlSimple


class ActorCriticModelPredictiveControlFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for policy and value function.
    It receives as input the observations and returns a tuple containing the features extracted for the policy
    and the features extracted for the value function.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer of the network.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        input_dim: int = 4,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, spaces.Box)

        self.input_dim = input_dim

        # We assume MlpPolicy
        # Extract features from input
        # Note: If you want to use images as input,
        # you will need to define a new CNN feature extractor.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_extractor(observations[..., : self.input_dim])


class ActorCriticModelPredictiveControlNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        mpc_class: Type[
            DistributionalModelPredictiveControlSimple
        ] = DistributionalModelPredictiveControlSimple,
        mpc_kwargs: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        predict_action: bool = False,
        predict_cost: bool = False,
        num_cost_terms: int = 1,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        assert predict_action != predict_cost, "Cannot predict both action and cost"

        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.predict_action = predict_action
        self.predict_cost = predict_cost
        self.num_cost_terms = num_cost_terms

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.ReLU(),
            nn.Linear(
                last_layer_dim_pi,
                action_dim * prediction_horizon if predict_action else num_cost_terms,
            ),
        )

        # MPC head of policy network
        self.policy_net_mpc = mpc_class(**mpc_kwargs)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(
        self, features: th.Tensor, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
                                        If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.forward_actor(features, obs), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, obs: th.Tensor) -> th.Tensor:
        # Defactorize obs
        agent_location = obs[..., 4:6]
        agent_velocity = obs[..., 6:8]
        target_location = obs[..., 8:10]
        target_velocity = obs[..., 10:12]

        with th.enable_grad():
            policy_net_output = self.policy_net(features)

            if self.predict_action:
                action_initial = policy_net_output.view(
                    (-1, self.prediction_horizon, self.action_dim)
                )

                action, _ = self.policy_net_mpc(
                    agent_location,
                    agent_velocity,
                    target_location,
                    target_velocity,
                    action_initial,
                    None,
                )
            else:
                cost_weights = policy_net_output.view((-1, self.num_cost_terms))
                cost_dict = {
                    "location_weight": cost_weights[..., 0],
                    "velocity_weight": cost_weights[..., 1],
                }
                action, _ = self.policy_net_mpc(
                    agent_location,
                    agent_velocity,
                    target_location,
                    target_velocity,
                    None,
                    cost_dict,
                )

        return action[:, 0]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ActorCriticModelPredictiveControlPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mpc_class: Type[
            DistributionalModelPredictiveControlSimple
        ] = DistributionalModelPredictiveControlSimple,
        mpc_kwargs: Optional[Dict[str, Any]] = None,
        predict_action: bool = False,
        predict_cost: bool = False,
        num_cost_terms: int = 1,
        *args,
        **kwargs,
    ):
        self.mpc_class = mpc_class
        self.mpc_kwargs = mpc_kwargs or {}
        self.action_dim = mpc_kwargs["action_size"]
        self.predict_action = predict_action
        self.predict_cost = predict_cost
        self.num_cost_terms = num_cost_terms

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = ActorCriticModelPredictiveControlNetwork(
            self.features_dim,
            action_dim=self.action_dim,
            mpc_class=self.mpc_class,
            mpc_kwargs=self.mpc_kwargs,
            prediction_horizon=self.mpc_kwargs["prediction_horizon"],
            predict_action=self.predict_action,
            predict_cost=self.predict_cost,
            num_cost_terms=self.num_cost_terms,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi,
        #         latent_sde_dim=latent_dim_pi,
        #         log_std_init=self.log_std_init,
        #     )
        # elif isinstance(
        #     self.action_dist,
        #     (
        #         CategoricalDistribution,
        #         MultiCategoricalDistribution,
        #         BernoulliDistribution,
        #     ),
        # ):
        #     self.action_net = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi
        #     )
        # else:
        #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            mean_actions, latent_vf = self.mlp_extractor(features, obs)
        else:
            pi_features, vf_features = features
            # latent_pi = self.mlp_extractor.forward_actor(pi_features)
            mean_actions = self.mlp_extractor.forward_actor(pi_features, obs)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        distribution = DiagGaussianDistribution(action_dim=self.action_dim)
        distribution = distribution.proba_distribution(
            mean_actions=mean_actions, log_std=th.zeros_like(mean_actions)
        )
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, self.action_dim))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features, obs)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, obs)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        distribution = DiagGaussianDistribution(action_dim=self.action_dim)
        distribution = distribution.proba_distribution(
            mean_actions=latent_pi, log_std=th.zeros_like(latent_pi)
        )
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Convert to torch tensor if needed
        observation = th.as_tensor(observation).to(self.device)

        # Compute the values
        with th.no_grad():
            action, _, _ = self.forward(observation, deterministic=deterministic)
            action = action.cpu().numpy()

        return action, state
