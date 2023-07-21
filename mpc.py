import torch
import torchopt
from torch import nn
import pygame
import utils

from stable_baselines3.common.distributions import DiagGaussianDistribution

from typing import Optional

pygame.font.init()
torch.autograd.set_detect_anomaly(True)


class ActionNetwork(nn.Module):
    def __init__(self, action: Optional[torch.Tensor] = None):
        super(ActionNetwork, self).__init__()

        if action is None:
            self.action = torch.zeros(2)
        else:
            self.action = nn.Parameter(action, requires_grad=True)

    def forward(self, k: int) -> torch.Tensor:
        return self.action[k]

    def __getitem__(self, index):
        return self.action[index]

    def get_action(self):
        return self.action

    def set_action(self, action: Optional[torch.Tensor] = None):
        if action is not None:
            self.action = nn.Parameter(action, requires_grad=True)


class ModelPredictiveControl(nn.Module):
    def __init__(
        self,
        system,
        action_size=2,
        control_horizon=1,
        prediction_horizon=10,
        num_optimization_step=40,
        lr=1e-2,
        size=10,
        window_size=512,
        agent_location_noise_level=0.05,
        agent_velocity_noise_level=0.05,
        target_location_noise_level=0.05,
        target_velocity_noise_level=0.05,
        location_weight=0.0,
        force_change_weight=0.0,
    ) -> None:
        super(ModelPredictiveControl, self).__init__()
        self.system = system
        self.action_size = action_size
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon
        self.num_optimization_step = num_optimization_step
        self.lr = lr
        self.size = size
        self.window_size = window_size
        self.agent_location_noise_level = agent_location_noise_level
        self.agent_velocity_noise_level = agent_velocity_noise_level
        self.target_location_noise_level = target_location_noise_level
        self.target_velocity_noise_level = target_velocity_noise_level

    def forward(
        self,
        agent_location: torch.Tensor,
        agent_velocity: torch.Tensor,
        target_location: torch.Tensor,
        target_velocity: torch.Tensor,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
        """

        self.reset(action_initial=action_initial)

        self._agent_location_original = agent_location
        self._agent_velocity_original = agent_velocity
        self._target_location_original = target_location
        self._target_velocity_original = target_velocity

        # Add noise to the observation
        self._agent_location = agent_location + (
            torch.randn_like(agent_location) * self.agent_location_noise_level
        )
        self._agent_velocity = agent_velocity + (
            torch.randn_like(agent_velocity) * self.agent_velocity_noise_level
        )
        self._target_location = target_location + (
            torch.randn_like(target_location) * self.target_location_noise_level
        )
        self._target_velocity = target_velocity + (
            torch.randn_like(target_velocity) * self.target_velocity_noise_level
        )

        return self._optimize(
            agent_location=self._agent_location,
            agent_velocity=self._agent_velocity,
            target_location=self._target_location,
            target_velocity=self._target_velocity,
            cost_dict=cost_dict,
        )

    def _optimize(
        self,
        agent_location,
        agent_velocity,
        target_location,
        target_velocity,
        cost_dict: Optional[dict] = None,
    ) -> None:
        """
        Optimizes the model.
        """
        for _ in range(self.num_optimization_step):
            self.optimizer.zero_grad()

            (
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
            ) = self._predict(
                agent_location, agent_velocity, target_location, target_velocity
            )
            loss = self._loss(
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
                cost_dict,
            )
            self.loss_value = loss
            (loss).backward(retain_graph=True, create_graph=True)
            # self.action.grad = torch.autograd.grad(
            #     loss,
            #     self.action,
            #     retain_graph=True,
            #     create_graph=True,
            # )[0]
            self.optimizer.step()
            self.action.retain_grad()

        action = self.action
        return action, loss

    def _predict(
        self, agent_location, agent_velocity, target_location, target_velocity
    ) -> torch.Tensor:
        predicted_agent_location = torch.zeros((self.prediction_horizon, 2))
        predicted_agent_velocity = torch.zeros((self.prediction_horizon, 2))
        predicted_target_location = torch.zeros((self.prediction_horizon, 2))
        predicted_target_velocity = torch.zeros((self.prediction_horizon, 2))

        for i in range(self.prediction_horizon):
            (
                agent_location,
                agent_velocity,
                target_location,
                target_velocity,
            ) = self.system(
                agent_location,
                agent_velocity,
                target_location,
                target_velocity,
                self.action[i],
            )
            predicted_agent_location[i] = agent_location
            predicted_agent_velocity[i] = agent_velocity
            predicted_target_location[i] = target_location
            predicted_target_velocity[i] = target_velocity

        return (
            predicted_agent_location,
            predicted_agent_velocity,
            predicted_target_location,
            predicted_target_velocity,
        )

    def _loss(
        self,
        agent_location,
        agent_velocity,
        target_location,
        target_velocity,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        # Calculate the loss
        # predicted_state -> (prediction_horizon, 4)
        # target -> (4,)

        assert cost_dict is not None, "cost_dict is None"

        # Calculate the distance
        location_loss = torch.norm(agent_location - target_location, 2, -1).mean()
        action_loss = self.action.sum()

        loss = (
            location_loss * cost_dict["location_weight"]
            + action_loss * cost_dict["action_weight"]
        )

        return loss

    def reset(self, action_initial: Optional[torch.Tensor] = None) -> None:
        """
        Resets the model.
        """
        if action_initial is None:
            action_initial = torch.zeros(
                (self.prediction_horizon, self.action_size), requires_grad=True
            )

        self.action = action_initial
        self.optimizer = torch.optim.SGD([self.action], lr=self.lr)


class ModelPredictiveControlSimple(nn.Module):
    def __init__(
        self,
        system,
        action_size=2,
        control_horizon=1,
        prediction_horizon=10,
        num_optimization_step=40,
        lr=1e-2,
        size=10,
        window_size=512,
        agent_location_noise_level=0.05,
        agent_velocity_noise_level=0.05,
        target_location_noise_level=0.05,
        target_velocity_noise_level=0.05,
    ) -> None:
        super(ModelPredictiveControlSimple, self).__init__()
        self.system = system
        self.action_size = action_size
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon
        self.num_optimization_step = num_optimization_step
        self.lr = lr
        self.size = size
        self.window_size = window_size
        self.agent_location_noise_level = agent_location_noise_level
        self.agent_velocity_noise_level = agent_velocity_noise_level
        self.target_location_noise_level = target_location_noise_level
        self.target_velocity_noise_level = target_velocity_noise_level

    def forward(
        self,
        agent_location: torch.Tensor,
        agent_velocity: torch.Tensor,
        target_location: torch.Tensor,
        target_velocity: torch.Tensor,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
        """

        self._agent_location_original = agent_location
        self._agent_velocity_original = agent_velocity
        self._target_location_original = target_location
        self._target_velocity_original = target_velocity

        # Add noise to the observation
        self._agent_location = agent_location + (
            torch.randn_like(agent_location) * self.agent_location_noise_level
        )
        self._agent_velocity = agent_velocity + (
            torch.randn_like(agent_velocity) * self.agent_velocity_noise_level
        )
        self._target_location = target_location + (
            torch.randn_like(target_location) * self.target_location_noise_level
        )
        self._target_velocity = target_velocity + (
            torch.randn_like(target_velocity) * self.target_velocity_noise_level
        )

        return self._optimize(
            agent_location=self._agent_location,
            agent_velocity=self._agent_velocity,
            target_location=self._target_location,
            target_velocity=self._target_velocity,
            action_initial=action_initial,
            cost_dict=cost_dict,
        )

    def _optimize(
        self,
        agent_location,
        agent_velocity,
        target_location,
        target_velocity,
        action_initial,
        cost_dict: Optional[dict] = None,
    ) -> None:
        """
        Optimizes the model.
        """

        batch_size = agent_location.shape[0]

        self._reset(action_initial, batch_size)

        for _ in range(self.num_optimization_step):
            (
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
            ) = self._predict(
                agent_location, agent_velocity, target_location, target_velocity
            )
            loss = self._loss(
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
                cost_dict,
            )
            self.loss_value = loss

            action_grad = torch.autograd.grad(
                loss,
                self.action,
                retain_graph=True,
                create_graph=True,
            )[0]

            self.action = self.action - self.lr * action_grad
            self.action.retain_grad()

        # self._predicted_agent_location = predicted_agent_location.detach()
        # self._predicted_agent_velocity = predicted_agent_velocity.detach()
        # self._predicted_target_location = predicted_target_location.detach()
        # self._predicted_target_velocity = predicted_target_velocity.detach()

        action = self.action  # .detach()

        return action, loss

    def _reset(
        self, action_initial: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> None:
        if action_initial is None:
            self.action = torch.zeros(
                (batch_size, self.prediction_horizon, self.action_size),
                requires_grad=True,
            )
        else:
            self.action = action_initial.clone()

    def _predict(
        self, agent_location, agent_velocity, target_location, target_velocity
    ) -> torch.Tensor:
        batch_size = agent_location.shape[0]

        assert (
            batch_size == self.action.shape[0]
        ), f"Input batch size is {batch_size}. Expected batch size is {self.action.shape[0]}."

        predicted_agent_location = torch.zeros((batch_size, self.prediction_horizon, 2))
        predicted_agent_velocity = torch.zeros((batch_size, self.prediction_horizon, 2))
        predicted_target_location = torch.zeros(
            (batch_size, self.prediction_horizon, 2)
        )
        predicted_target_velocity = torch.zeros(
            (batch_size, self.prediction_horizon, 2)
        )

        for i in range(self.prediction_horizon):
            (
                agent_location,
                agent_velocity,
                target_location,
                target_velocity,
            ) = self.system(
                agent_location,
                agent_velocity,
                target_location,
                target_velocity,
                self.action[:, i],
            )
            predicted_agent_location[:, i] = agent_location
            predicted_agent_velocity[:, i] = agent_velocity
            predicted_target_location[:, i] = target_location
            predicted_target_velocity[:, i] = target_velocity

        return (
            predicted_agent_location,
            predicted_agent_velocity,
            predicted_target_location,
            predicted_target_velocity,
        )

    def _loss(
        self,
        agent_location,
        agent_velocity,
        target_location,
        target_velocity,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        # Calculate the loss
        # predicted_state -> (prediction_horizon, 4)
        # target -> (4,)

        if cost_dict is None:
            cost_dict = {
                "location_weight": 1.0,
                "velocity_weight": 0.1,
            }

        # Calculate the distance
        location_loss = torch.norm(agent_location - target_location, 2, -1).mean(dim=1)
        velocity_loss = torch.norm(agent_velocity - target_velocity, 2, -1).mean(dim=1)

        # action_loss = self.action.sum(dim=1)

        loss = (location_loss * cost_dict["location_weight"]).mean() + (
            velocity_loss * cost_dict["velocity_weight"]
        ).mean()

        return loss


class MetaModelPredictiveControl(ModelPredictiveControl):
    def __init__(
        self,
        system,
        action_size=2,
        control_horizon=1,
        prediction_horizon=10,
        num_optimization_step=40,
        lr=1e-2,
        size=10,
        window_size=512,
        agent_location_noise_level=0.05,
        agent_velocity_noise_level=0.05,
        target_location_noise_level=0.05,
        target_velocity_noise_level=0.05,
        location_weight=0.0,
        force_change_weight=0.0,
    ) -> None:
        super(MetaModelPredictiveControl, self).__init__(
            system,
            action_size,
            control_horizon,
            prediction_horizon,
            num_optimization_step,
            lr,
            size,
            window_size,
            agent_location_noise_level,
            agent_velocity_noise_level,
            target_location_noise_level,
            target_velocity_noise_level,
            location_weight,
            force_change_weight,
        )

        action = torch.zeros((self.prediction_horizon, self.action_size))
        self.action = ActionNetwork(action=action)
        self.optimizer = torchopt.MetaAdam(self.action, lr=self.lr)

    def _optimize(
        self, agent_location, agent_velocity, target_location, target_velocity
    ) -> None:
        """
        Optimizes the model.
        """
        for _ in range(self.num_optimization_step):
            # self.optimizer.zero_grad()

            (
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
            ) = self._predict(
                agent_location, agent_velocity, target_location, target_velocity
            )
            loss = self._loss(
                predicted_agent_location,
                predicted_agent_velocity,
                predicted_target_location,
                predicted_target_velocity,
                # self._target_location_original,
                # self._target_velocity_original,
            )
            self.loss_value = loss
            # loss.backward(retain_graph=True)
            self.optimizer.step(loss)

        self._predicted_agent_location = predicted_agent_location.detach()
        self._predicted_agent_velocity = predicted_agent_velocity.detach()
        self._predicted_target_location = predicted_target_location.detach()
        self._predicted_target_velocity = predicted_target_velocity.detach()

        action = self.action.get_action()  # .detach()

        return action, loss

    def reset(self, action_initial: Optional[torch.Tensor] = None) -> None:
        if action_initial is None:
            action_initial = torch.zeros((self.prediction_horizon, self.action_size))

        self.action = ActionNetwork(action=action_initial)
        self.optimizer = torchopt.MetaAdam(self.action, lr=self.lr)


class DistributionalModelPredictiveControlSimple(ModelPredictiveControlSimple):
    def __init__(
        self,
        system,
        action_size=2,
        control_horizon=1,
        prediction_horizon=10,
        num_optimization_step=40,
        lr=1e-2,
        size=10,
        window_size=512,
        agent_location_noise_level=0.05,
        agent_velocity_noise_level=0.05,
        target_location_noise_level=0.05,
        target_velocity_noise_level=0.05,
    ) -> None:
        super(DistributionalModelPredictiveControlSimple, self).__init__(
            system,
            action_size,
            control_horizon,
            prediction_horizon,
            num_optimization_step,
            lr,
            size,
            window_size,
            agent_location_noise_level,
            agent_velocity_noise_level,
            target_location_noise_level,
            target_velocity_noise_level,
        )

    def _reset(self, action_initial: torch.Tensor):
        if action_initial is None:
            action_initial = torch.zeros(
                (self.batch_size, self.prediction_horizon, self.action_size),
                requires_grad=True,
            )

        distribution = DiagGaussianDistribution(self.action_size)
        distribution.proba_distribution(
            mean_actions=action_initial.clone(),
            log_std=torch.zeros_like(action_initial),
        )

        self.action = distribution.sample()
