import env
from mpc import (
    ModelPredictiveControl,
    MetaModelPredictiveControl,
    ModelPredictiveControlSimple,
)
from system import DynamicalSystem
from gymnasium.wrappers import FlattenObservation

import gymnasium as gym

import torch
from torch import nn


class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()

        self.fc1 = nn.Linear(8, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


def main():
    size = 10
    agent_location_noise_level = 0.0
    agent_velocity_noise_level = 0.0
    target_location_noise_level = 0.0
    target_velocity_noise_level = 0.0

    # Create system
    system = DynamicalSystem(
        size=size,
        random_force_probability=0.00,
        random_force_magnitude=10.0,
        friction_coefficient=0.1,
        wind_gust=[0.5, 0.5],
        wind_gust_region=[[0.3, 0.7], [0.3, 0.7]],
    )

    # Create environment
    env = gym.make(
        "DynamicalSystem-v0",
        render_mode="rgb_array",
        size=size,
        distance_threshold=0.5,
        system=system,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
        force_penalty_level=0.1,
    )
    env = FlattenObservation(env)
    observation, _ = env.reset()

    # Create Model Predictive Control model
    mpc = ModelPredictiveControlSimple(
        system,
        size=size,
        lr=0.5,
        prediction_horizon=10,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
        num_optimization_step=20,
    )

    dummy_network = DummyNetwork()
    dummy_network.train()

    observation_tensor = torch.Tensor(observation).unsqueeze(0)

    agent_location_ = observation_tensor[:, :2]
    agent_velocity_ = observation_tensor[:, 2:4]
    target_location_ = observation_tensor[:, 4:6]
    target_velocity_ = observation_tensor[:, 6:8]

    target_location_ = agent_location_.clone() + torch.Tensor([0.5, 0.5])

    optimizer = torch.optim.Adam(
        [
            {"params": dummy_network.parameters()},
        ],
        lr=1e-2,
    )

    while True:
        optimizer.zero_grad()

        action_initial = dummy_network(observation_tensor)
        action_initial = action_initial.view(1, 10, 2)
        cost_dict = {
            "location_weight": 1.0,
            "action_weight": 0.0,
        }

        print(cost_dict)

        action, loss_inner = mpc(
            agent_location_,
            agent_velocity_,
            target_location_,
            target_velocity_,
            action_initial=action_initial,
            cost_dict=cost_dict,
        )

        print(action[0])

        (agent_location, agent_velocity, target_location, target_velocity) = system(
            agent_location_,
            agent_velocity_,
            target_location_,
            target_velocity_,
            action[0],
        )

        loss_outer = torch.norm(target_location - agent_location, 2, -1)
        # loss_outer = torch.norm(action.sum(), 2, -1)

        print(loss_outer.item())

        loss_outer.backward()

        print(list(dummy_network.parameters())[0].grad)
        print(list(dummy_network.parameters())[1].grad)

        optimizer.step()


if __name__ == "__main__":
    main()
