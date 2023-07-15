from mpc import ModelPredictiveControl
from system import DynamicalSystem

import numpy as np
import torch
from torch import nn

size = 20

observation = torch.cat(
	[
		torch.rand(2) * size,
		torch.rand(2) * 1

	]
)
target = torch.cat(
	[
		torch.rand(2) * size,
		torch.zeros(2)
	]
)

system = DynamicalSystem()
mpc = ModelPredictiveControl(system, size = size)



mpc.reset()

while True:
    
	action = mpc(observation, target)
	print(action[0])

	observation = system(observation, action[0])
	print(observation)
	
	mpc.render()
	mpc.reset()