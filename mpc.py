import torch
import torchopt
from torch import nn
import pygame
from linearizer import GymEnvironmentLinearizer
import utils
import gymnasium as gym
from typing import Tuple, Optional, Any, List, Union
from copy import deepcopy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.distributions import DiagGaussianDistribution

from typing import Optional, Tuple, Callable

pygame.font.init()


class ModelPredictiveControlWithoutOptimizer(nn.Module):
    def __init__(
        self,
        system: nn.Module,
        cost: nn.Module,
        action_size: int = 2,
        prediction_horizon: int = 10,
        num_optimization_step: int = 40,
        lr: float = 1e-2,
        std: float = 0.3,
        device="cuda",
    ) -> None:
        super(ModelPredictiveControlWithoutOptimizer, self).__init__()
        self.system = system
        self.cost = cost
        self.action_size = action_size
        self.prediction_horizon = prediction_horizon
        self.num_optimization_step = num_optimization_step
        self.lr = lr
        self.std = std

        self.device = device

    def forward(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
        """
        return self._optimize(
            current_state=current_state,
            target_state=target_state,
            action_initial=action_initial,
            cost_dict=cost_dict,
        )

    def _optimize(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimizes the model.
        """

        if action_initial is None:
            batch_size = list(current_state.values())[0].shape[0]
        else:
            batch_size = action_initial.shape[0]

        self._reset(action_initial, batch_size)

        loss = None

        for _ in range(self.num_optimization_step):
            # self.action.grad = None

            (predicted_state) = self._predict(current_state)

            loss = self._loss(
                predicted_state,
                target_state,
                cost_dict,
            )

            # loss.backward(retain_graph=True, create_graph=True, inputs=(self.action,))
            action_grad = torch.autograd.grad(
                loss,
                self.action,
                retain_graph=True,
                create_graph=True,
            )[0]

            # torch.nn.utils.clip_grad_norm_(self.action, 1.0)

            # self.action = self.action - self.lr * self.action.grad
            self.action = self.action - self.lr * action_grad
            # self.action.retain_grad()

        action = self.action  # .detach()

        return action, loss

    def _reset(
        self, action_initial: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> None:
        if action_initial is None:
            # self.action = torch.zeros(
            #     (batch_size, self.prediction_horizon, self.action_size),
            #     device=self.device,
            #     requires_grad=True,
            # )
            self.action = (
                torch.randn(
                    (batch_size, self.prediction_horizon, self.action_size),
                    device=self.device,
                    requires_grad=True,
                )
                * self.std
            )

        else:
            self.action = action_initial

        # self.action.register_hook(lambda grad: print(grad))

    def _predict(self, state: dict) -> torch.Tensor:
        predicted_state_ = list()

        for i in range(self.prediction_horizon):
            (state) = self.system(
                state,
                self.action[:, i],
            )
            predicted_state_.append(state)

        predicted_state = dict()
        elem = predicted_state_[0]
        for k in elem.keys():
            predicted_state[k] = torch.stack([x[k] for x in predicted_state_], dim=1)

        return predicted_state

    def _loss(
        self,
        predicted_state: dict,
        target_state: dict,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        return self.cost(predicted_state, target_state, self.action, cost_dict)


class EnvironmentPredictiveControlWithoutOptimizer(nn.Module):
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        cost: nn.Module,
        action_size: int = 2,
        prediction_horizon: int = 10,
        num_optimization_step: int = 40,
        lr: float = 1e-2,
        std: float = 0.3,
        device="cuda",
    ) -> None:
        super(EnvironmentPredictiveControlWithoutOptimizer, self).__init__()
        self.env = env
        self.cost = cost
        self.action_size = action_size
        self.prediction_horizon = prediction_horizon
        self.num_optimization_step = num_optimization_step
        self.lr = lr
        self.std = std
        self.linearizer = GymEnvironmentLinearizer(
            env=self.env, eps=0.5, state_dynamics=True
        )
        self.device = device

    def forward(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
        """
        return self._optimize(
            current_state=current_state,
            target_state=target_state,
            action_initial=action_initial,
            cost_dict=cost_dict,
        )

    def _optimize(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimizes the model.
        """

        if action_initial is None:
            batch_size = current_state.shape[0]
        else:
            batch_size = action_initial.shape[0]

        self._reset(action_initial, batch_size)

        loss = None

        for _ in range(self.num_optimization_step):
            # self.action.grad = None

            (predicted_state) = self._predict(current_state)

            loss = self._loss(
                predicted_state,
                target_state,
                cost_dict,
            )
            print(f"Loss: {loss}")
            # loss.backward(retain_graph=True, create_graph=True, inputs=(self.action,))
            action_grad = torch.autograd.grad(
                loss,
                self.action,
                retain_graph=True,
                create_graph=True,
            )[0]
            print(f"Action grad: {action_grad.sum()}")
            # torch.nn.utils.clip_grad_norm_(self.action, 1.0)
            
            # self.action = self.action - self.lr * self.action.grad
            self.action = self.action - self.lr * action_grad
            # self.action.retain_grad()
        
        print(self.action)
        action = self.action  # .detach()

        return action, loss

    def _reset(
        self, action_initial: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> None:
        if action_initial is None:
            # self.action = torch.zeros(
            #     (batch_size, self.prediction_horizon, self.action_size),
            #     device=self.device,
            #     requires_grad=True,
            # )
            self.action = (
                torch.randn(
                    (batch_size, self.prediction_horizon, self.action_size),
                    device=self.device,
                    requires_grad=True,
                )
                * self.std
            )

        else:
            self.action = action_initial

        # self.action.register_hook(lambda grad: print(grad))

    def _predict(self, state) -> torch.Tensor:
        predicted_state_ = list()

        action_operating_point = self.action[:, 0].clone().detach().numpy()
        state_operating_point = state.clone().detach().numpy()

        state_dynamics = self.linearizer(state_operating_point, action_operating_point)

        delta_state = torch.zeros_like(state).unsqueeze(-1)
        delta_action = self.action - self.action[:, 0]

        print(state_dynamics.b_matrix)

        for i in range(0, self.prediction_horizon):
            (state) = state_dynamics(
                delta_state,
                delta_action[:, i : i + 1],
            )
            predicted_state_.append(state)

        predicted_state = torch.stack(predicted_state_, dim=1).squeeze(-1)

        return predicted_state

    def _loss(
        self,
        predicted_state: dict,
        target_state: dict,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        return self.cost(predicted_state, target_state, self.action, cost_dict)


# class ModelPredictiveControlWithOptimizer(nn.Module):
#     def __init__(
#         self,
#         system,
#         action_size=2,
#         control_horizon=1,
#         prediction_horizon=10,
#         num_optimization_step=40,
#         lr=1e-2,
#     ) -> None:
#         super(ModelPredictiveControlWithOptimizer, self).__init__()
#         self.system = system
#         self.action_size = action_size
#         self.control_horizon = control_horizon
#         self.prediction_horizon = prediction_horizon
#         self.num_optimization_step = num_optimization_step
#         self.lr = lr

#     def forward(
#         self,
#         agent_location: torch.Tensor,
#         agent_velocity: torch.Tensor,
#         target_location: torch.Tensor,
#         target_velocity: torch.Tensor,
#         action_initial: Optional[torch.Tensor] = None,
#         cost_dict: Optional[dict] = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
#         Returns:
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
#         """

#         self.reset(action_initial=action_initial)

#         self._agent_location = agent_location
#         self._agent_velocity = agent_velocity
#         self._target_location = target_location
#         self._target_velocity = target_velocity

#         return self._optimize(
#             agent_location=self._agent_location,
#             agent_velocity=self._agent_velocity,
#             target_location=self._target_location,
#             target_velocity=self._target_velocity,
#             cost_dict=cost_dict,
#         )

#     def _optimize(
#         self,
#         agent_location,
#         agent_velocity,
#         target_location,
#         target_velocity,
#         cost_dict: Optional[dict] = None,
#     ) -> None:
#         """
#         Optimizes the model.
#         """
#         for _ in range(self.num_optimization_step):
#             self.optimizer.zero_grad()

#             (
#                 predicted_agent_location,
#                 predicted_agent_velocity,
#                 predicted_target_location,
#                 predicted_target_velocity,
#             ) = self._predict(
#                 agent_location, agent_velocity, target_location, target_velocity
#             )
#             loss = self._loss(
#                 predicted_agent_location,
#                 predicted_agent_velocity,
#                 predicted_target_location,
#                 predicted_target_velocity,
#                 cost_dict,
#             )
#             self.loss_value = loss
#             (loss).backward(retain_graph=True, create_graph=True)
#             # self.action.grad = torch.autograd.grad(
#             #     loss,
#             #     self.action,
#             #     retain_graph=True,
#             #     create_graph=True,
#             # )[0]
#             self.optimizer.step()
#             self.action.retain_grad()

#         action = self.action
#         return action, loss

#     def _predict(
#         self, agent_location, agent_velocity, target_location, target_velocity
#     ) -> torch.Tensor:
#         predicted_agent_location = torch.zeros((self.prediction_horizon, 2))
#         predicted_agent_velocity = torch.zeros((self.prediction_horizon, 2))
#         predicted_target_location = torch.zeros((self.prediction_horizon, 2))
#         predicted_target_velocity = torch.zeros((self.prediction_horizon, 2))

#         for i in range(self.prediction_horizon):
#             (
#                 agent_location,
#                 agent_velocity,
#                 target_location,
#                 target_velocity,
#             ) = self.system(
#                 agent_location,
#                 agent_velocity,
#                 target_location,
#                 target_velocity,
#                 self.action[i],
#             )
#             predicted_agent_location[i] = agent_location
#             predicted_agent_velocity[i] = agent_velocity
#             predicted_target_location[i] = target_location
#             predicted_target_velocity[i] = target_velocity

#         return (
#             predicted_agent_location,
#             predicted_agent_velocity,
#             predicted_target_location,
#             predicted_target_velocity,
#         )

#     def _loss(
#         self,
#         agent_location,
#         agent_velocity,
#         target_location,
#         target_velocity,
#         cost_dict: Optional[dict] = None,
#     ) -> torch.Tensor:
#         # Calculate the loss
#         # predicted_state -> (prediction_horizon, 4)
#         # target -> (4,)

#         assert cost_dict is not None, "cost_dict is None"

#         # Calculate the distance
#         location_loss = torch.norm(agent_location - target_location, 2, -1).mean()
#         action_loss = self.action.sum()

#         loss = (
#             location_loss * cost_dict["location_weight"]
#             + action_loss * cost_dict["action_weight"]
#         )

#         return loss

#     def reset(self, action_initial: Optional[torch.Tensor] = None) -> None:
#         """
#         Resets the model.
#         """
#         if action_initial is None:
#             action_initial = torch.zeros(
#                 (self.prediction_horizon, self.action_size), requires_grad=True
#             )

#         self.action = action_initial
#         self.optimizer = torch.optim.SGD([self.action], lr=self.lr)


# class DistributionalModelPredictiveControlSimple(ModelPredictiveControlSimple):
#     def __init__(
#         self,
#         system,
#         action_size=2,
#         control_horizon=1,
#         prediction_horizon=10,
#         num_optimization_step=40,
#         lr=1e-2,
#     ) -> None:
#         super(DistributionalModelPredictiveControlSimple, self).__init__(
#             system,
#             action_size,
#             control_horizon,
#             prediction_horizon,
#             num_optimization_step,
#             lr,
#         )

#     def _reset(self, action_initial: torch.Tensor, batch_size: int = 1) -> None:
#         if action_initial is None:
#             action_initial = torch.zeros(
#                 (batch_size, self.prediction_horizon, self.action_size),
#                 requires_grad=True,
#             )

#         distribution = DiagGaussianDistribution(self.action_size)
#         distribution = distribution.proba_distribution(
#             mean_actions=action_initial.clone(),
#             log_std=torch.zeros_like(action_initial, requires_grad=True),
#         )

#         self.action = distribution.sample()
