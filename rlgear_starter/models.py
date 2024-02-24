from typing import Any
import gymnasium as gym
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np
import rlgear.models
from torch import Tensor
import torch.nn
from rlgear.rllib_utils import check


# pylint: disable=abstract-method
class FCNet(rlgear.models.TorchModel):  # type: ignore
    def __init__(
        self,
        obs_space: gym.Space[Any],
        action_space: gym.Space[Any],
        num_outputs: int,
        model_config: dict[str, Any],
        name: str,
        offset: float,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # this variable just exists to show how to input a variable to a custom model
        self.offset = offset

        assert obs_space.shape is not None
        num_inp = int(np.prod(obs_space.shape))

        def _make_net(_num_inp: int, _num_out: int) -> rlgear.models.MLPNet:
            hiddens = model_config["fcnet_hiddens"]
            return rlgear.models.MLPNet(num_inp, _num_out, hiddens, 0.0, torch.nn.ReLU)

        self.value_net = _make_net(num_inp, 1)
        self.policy_net = _make_net(num_inp, num_outputs)

    # pylint: disable=unused-argument
    @override(TorchModelV2)  # type: ignore
    def forward(
        self, input_dict: dict[str, Tensor], state: list[Tensor], seq_lens: Tensor
    ) -> tuple[Tensor, list[Tensor]]:

        obs = input_dict["obs"]
        self._cur_value = self.value_net(obs).squeeze(-1)
        self._last_output = self.policy_net(obs)

        check(self._last_output, obs=obs, val=self._cur_value)
        check(self._cur_value, obs=obs, logits=self._last_output)
        return self._last_output, []
