from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

from gymnasium import spaces
import torch as th
from torch import nn
import numpy as np
from functools import partial
import time

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
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback
from mpc import ModelPredictiveControlWithoutOptimizer


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
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(input_dim, features_dim),
        #     nn.ReLU(),
        #     nn.Linear(features_dim, features_dim),
        #     nn.ReLU(),
        #     nn.Linear(features_dim, features_dim),
        # )

        # Flatten feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_extractor(observations[..., : self.input_dim])


class ActorCriticModelPredictiveControlPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mpc_class: Type[
            ModelPredictiveControlWithoutOptimizer
        ] = ModelPredictiveControlWithoutOptimizer,
        mpc_kwargs: Optional[Dict[str, Any]] = None,
        predict_action: bool = False,
        predict_cost: bool = False,
        num_cost_terms: int = 1,
        obs_to_state_target: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        self.action_dim = mpc_kwargs["action_size"]
        self.prediction_horizon = mpc_kwargs["prediction_horizon"]
        self.predict_action = predict_action
        self.predict_cost = predict_cost
        self.num_cost_terms = num_cost_terms
        self.obs_to_state_target = obs_to_state_target

        assert (
            self.predict_action != self.predict_cost
        ), "Either predict action or predict cost must be True"

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[64, 64], vf=[64, 64])

        kwargs["net_arch"]["pi"].append(
            self.prediction_horizon * self.action_dim
            if self.predict_action
            else self.prediction_horizon * self.num_cost_terms
        )

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

        self.mpc = mpc_class(**mpc_kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

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
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        with th.enable_grad():
            if self.predict_action:
                action_initial = latent_pi.view(
                    batch_size, self.prediction_horizon, self.action_dim
                )
                action_initial.requires_grad = True
                mean_actions, _ = self.mpc(state, target, action_initial, None)

            else:
                cost_dict = latent_pi.view(
                    batch_size, self.prediction_horizon, self.num_cost_terms
                )

                mean_actions, _ = self.mpc(state, target, None, cost_dict)

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate the values, log probability and entropy for given actions given the current policy

        :param obs: Observation
        :param actions: Actions
        :return: values, log probability and entropy
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        if self.predict_action:
            action_initial = latent_pi.view(
                batch_size, self.prediction_horizon, self.action_dim
            )

            mean_actions, _ = self.mpc(state, target, action_initial, None)

        else:
            cost_dict = latent_pi.view(
                batch_size, self.prediction_horizon, self.num_cost_terms
            )

            mean_actions, _ = self.mpc(state, target, None, cost_dict)

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

        log_prob = distribution.log_prob(actions)
        dist_entropy = distribution.entropy()
        values = self.value_net(latent_vf).flatten()
        return values, log_prob, dist_entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        with th.enable_grad():
            if self.predict_action:
                action_initial = latent_pi.view(
                    batch_size, self.prediction_horizon, self.action_dim
                )
                action_initial.requires_grad = True
                mean_actions, _ = self.mpc(state, target, action_initial, None)

            else:
                cost_dict = latent_pi.view(
                    batch_size, self.prediction_horizon, self.num_cost_terms
                )

                cost_dict = th.clamp(cost_dict, min=10.0, max=10.0)

                mean_actions, _ = self.mpc(state, target, None, cost_dict)

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")


class ActorCriticModelPredictiveControlTeacherForcingPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mpc_class: Type[
            ModelPredictiveControlWithoutOptimizer
        ] = ModelPredictiveControlWithoutOptimizer,
        mpc_kwargs: Optional[Dict[str, Any]] = None,
        predict_action: bool = False,
        predict_cost: bool = False,
        num_cost_terms: int = 1,
        obs_to_state_target: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        self.action_dim = mpc_kwargs["action_size"]
        self.prediction_horizon = mpc_kwargs["prediction_horizon"]
        self.predict_action = predict_action
        self.predict_cost = predict_cost
        self.num_cost_terms = num_cost_terms
        self.obs_to_state_target = obs_to_state_target

        assert (
            self.predict_action != self.predict_cost
        ), "Either predict action or predict cost must be True"

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[64, 64], vf=[64, 64])

        kwargs["net_arch"]["pi"].append(
            self.prediction_horizon * self.action_dim
            if self.predict_action
            else self.prediction_horizon * self.num_cost_terms
        )

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

        self.mpc = mpc_class(**mpc_kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

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
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        with th.enable_grad():
            if self.predict_action:
                action_initial = latent_pi.view(
                    batch_size, self.prediction_horizon, self.action_dim
                )
                action_initial.requires_grad = True
                mean_actions, _ = self.mpc(state, target, action_initial, None)

            else:
                cost_dict = latent_pi.view(
                    batch_size, self.prediction_horizon, self.num_cost_terms
                )

                mean_actions, _ = self.mpc(state, target, None, cost_dict)

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate the values, log probability and entropy for given actions given the current policy

        :param obs: Observation
        :param actions: Actions
        :return: values, log probability and entropy
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        if self.predict_action:
            action_initial = latent_pi.view(
                batch_size, self.prediction_horizon, self.action_dim
            )

            # mean_actions, _ = self.mpc(state, target, action_initial, None)
            # Instead, we directly use the predicted action, which we will try to
            # make as close as possible to the more accurate MPC prediction.
            mean_actions = action_initial

        else:
            raise NotImplementedError

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

        log_prob = distribution.log_prob(actions)
        dist_entropy = distribution.entropy()
        values = self.value_net(latent_vf).flatten()
        return values, log_prob, dist_entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)

        state, target = self.obs_to_state_target(obs)
        batch_size = obs.shape[0]

        with th.enable_grad():
            if self.predict_action:
                action_initial = latent_pi.view(
                    batch_size, self.prediction_horizon, self.action_dim
                )
                action_initial.requires_grad = True
                mean_actions, _ = self.mpc(state, target, action_initial, None)

            else:
                cost_dict = latent_pi.view(
                    batch_size, self.prediction_horizon, self.num_cost_terms
                )

                cost_dict = th.clamp(cost_dict, min=10.0, max=10.0)

                mean_actions, _ = self.mpc(state, target, None, cost_dict)

        mean_actions = mean_actions[:, 0]

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")
