import torch
from torch import nn


class MappingNet(nn.Module):
    def __init__(
        self,
        input_nc=70,
        hidden_dim=256,
        output_dim=256,
        num_layers=4,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        num_layers = max(int(num_layers), 2)

        layers = [
            nn.Conv1d(input_nc, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=1))
        self.first = nn.Sequential(*layers)

    def forward(self, input_3dmm):
        # ---- channel-compat guard (70 vs 73 etc.) ----
        # Some SadTalker forks produce 73-dim 3DMM while this model expects 70.
        # To keep wrapper portable/non-breaking, adapt input channels here.
        try:
            expected_c = self.first[0].in_channels  # Conv1d in_channels
        except Exception:
            expected_c = input_3dmm.shape[1]

        cur_c = input_3dmm.shape[1]
        if cur_c != expected_c:
            # slice extra channels or pad missing channels with zeros
            if cur_c > expected_c:
                input_3dmm = input_3dmm[:, :expected_c, :]
            else:
                pad = input_3dmm.new_zeros(
                    (input_3dmm.shape[0], expected_c - cur_c, input_3dmm.shape[2])
                )
                input_3dmm = torch.cat([input_3dmm, pad], dim=1)
            print(f"[WARN] MappingNet: adapted input_3dmm channels {cur_c} -> {expected_c}")

        out = self.first(input_3dmm)
        return out
