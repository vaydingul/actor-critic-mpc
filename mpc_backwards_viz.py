import env
from mpc import ModelPredictiveControl, MetaModelPredictiveControl
from system import DynamicalSystem
from gymnasium.wrappers import FlattenObservation

import gymnasium as gym
from torchviz import make_dot
import torch
from torch import nn
import torchopt

torch.autograd.set_detect_anomaly(True)


class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()

        self.fc1 = nn.Linear(8, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 20)

        # Initialize all weight and biases as zero
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.fill_(0.0)
        #         m.bias.data.fill_(0.0)

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
    mpc = MetaModelPredictiveControl(
        system,
        size=size,
        lr=0.5,
        prediction_horizon=10,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
        num_optimization_step=20,
        location_weight=1.0,
        force_change_weight=0.0,
    )

    dummy_network = DummyNetwork()

    observation_tensor = torch.Tensor(observation).requires_grad_(True)

    agent_location_ = observation_tensor[:2]
    agent_velocity_ = observation_tensor[2:4]
    target_location_ = observation_tensor[4:6]
    target_velocity_ = observation_tensor[6:8]

    optimizer = torch.optim.Adam(dummy_network.parameters(), lr=0.1)

    optimizer.zero_grad()

    action_initial = dummy_network(observation_tensor)
    action_initial = action_initial.view(10, 2)

    net_state_0 = torchopt.extract_state_dict(
        mpc, enable_visual=True, visual_prefix="step0."
    )

    action, loss_inner = mpc(
        agent_location_,
        agent_velocity_,
        target_location_,
        target_velocity_,
        action_initial,
    )

    (agent_location, agent_velocity, target_location, target_velocity) = system(
        agent_location_,
        agent_velocity_,
        target_location_,
        target_velocity_,
        action[0],
    )

    net_state_1 = torchopt.extract_state_dict(
        mpc, enable_visual=True, visual_prefix="step1."
    )

    loss_outer = torch.norm(target_location - agent_location, 2, -1)

    print(loss_outer.item())

    viz = torchopt.visual.make_dot(
        loss_outer,
        params=(
            {"loss": loss_outer},
            dummy_network.named_parameters(),
            net_state_0,
            net_state_1,
        ),
        show_attrs=False,
        show_saved=False,
    ).render("mpc", format="png", view=True)

    # loss_outer.backward(retain_graph=True)

    # optimizer.step()


if __name__ == "__main__":
    main()


# loss = torch.norm(action_selected - action_target, 2, -1)

# loss.backward(retain_graph=True)

# optimizer.step()
