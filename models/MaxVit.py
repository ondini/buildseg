import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


class UpSamplingBlock(nn.Module):
    """
    A very basic Upsampling block (these params have to be learnt from scratch so keep them small)

    First it reduces the number of channels of the incoming layer to the amount of the skip connection with a 1x1 conv
    then it concatenates them and combines them in a new conv layer.



    x --> up ---> conv1 --> concat --> conv2 --> norm -> relu
                  ^
                  |
                  skip_x
    """

    def __init__(self, n_channels_in, n_skip_channels_in, n_channels_out, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_skip_channels_in + n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )

        self.norm1 = nn.BatchNorm2d(n_channels_out)
        self.relu1 = nn.ReLU()

    def forward(self, x, x_skip):
        # bilinear is not deterministic, use nearest neighbor instead
        x = nn.functional.interpolate(x, scale_factor=2.0)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # second conv as in original UNet upsampling block decreases performance
        # probably because I was using a small dataset that did not have enough data to learn the extra parameters
        return x


class MaxVitUnet(nn.Module):
    """
    Pretrained MaxVit(MBConv (Efficient Net) + Blocked Local Attention + Grid global attention) as Encoder for the U-Net.

    the outputs of the stem and all Multi-Axis (Max) stages are used as feature layers
    note that the paper uses only stage 2-4 for segmentation w/ Mask-RCNN.

    maxvit_nano_rw_256 is a version trained on 256x256 images in timm, that differs slightly from the paper
    but is a much more lightweight model (approx. 15M params)

    It is approx 4 times slower than the ConvNeXt femto backbone (5M params), and still
    about 2 times slower than convnext_nano @ 15M params, yet provided better results
    than both convnext variants in some initial experiments.

    The model can deal with input sizes divisible by 32, but for pretrained weights you are restricted to multiples of the pretrained
    models: 224, 256, 384. From the accompanying notebook, it seems that the model easily handles images that are 3 times as big as the
    training size. (see https://github.com/rwightman/pytorch-image-models/issues/1475 for more details)

    For now only 256 is supported so input sizes are restricted to 256,512,...



    orig                    ---   1/1  -->                       --->       (head)
        stem                ---   1/2  -->             decode4
            stage 1         ---   1/4  -->         decode3
                stage 2     ---   1/8  -->     decode2
                    stage 3 ---   1/16 --> decode1
                        stage 4 ---1/32----|
    """

    # 15M params
    FEATURE_CONFIG = [
        {"down": 2, "channels": 64},
        {"down": 4, "channels": 64},
        {"down": 8, "channels": 128},
        {"down": 16, "channels": 256},
        {"down": 32, "channels": 512},
    ]
    MODEL_NAME = "maxvit_nano_rw_256"
    feature_layers = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]

    def __init__(self, n_heatmaps=1, logits=True) -> None:
        super().__init__()
        self.encoder = timm.create_model(self.MODEL_NAME, pretrained=True, num_classes=0)
        self.feature_extractor = create_feature_extractor(self.encoder, self.feature_layers)
        self.decoder_blocks = nn.ModuleList()
        for config_skip, config_in in zip(self.FEATURE_CONFIG, self.FEATURE_CONFIG[1:]):
            block = UpSamplingBlock(config_in["channels"], config_skip["channels"], config_skip["channels"], 3)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(
            self.FEATURE_CONFIG[0]["channels"], self.FEATURE_CONFIG[0]["channels"], 3, padding="same"
        )
        self.final_upsampling_block = UpSamplingBlock(
            self.FEATURE_CONFIG[0]["channels"], 3, self.FEATURE_CONFIG[0]["channels"], 3
        )
        
        self.head = nn.Conv2d(
            in_channels=self.FEATURE_CONFIG[0]["channels"],
            out_channels=n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )

        # expect output of backbone to be normalized!
        # so by filling bias to -4, the sigmoid should be on avg sigmoid(-4) =  0.02
        # which is consistent with the desired heatmaps that are zero almost everywhere.
        # setting too low would result in loss of gradients..
        self.head.bias.data.fill_(-4)

        self.logits = logits

    def forward(self, x):
        orig_x = torch.clone(x)
        features = list(self.feature_extractor(x).values())
        x = features.pop(-1)
        for block in self.decoder_blocks[::-1]:
            x = block(x, features.pop(-1))

        # x = nn.functional.interpolate(x, scale_factor=2)
        # x = self.final_conv(x)
        x = self.final_upsampling_block(x, orig_x)
        out = self.head(x)
        return torch.sigmoid(out) if not self.logits else out

        
    


class MaxVitPicoUnet(MaxVitUnet):
    MODEL_NAME = "maxvit_rmlp_pico_rw_256"  # 7.5M params.
    FEATURE_CONFIG = [
        {"down": 2, "channels": 32},
        {"down": 4, "channels": 32},
        {"down": 8, "channels": 64},
        {"down": 16, "channels": 128},
        {"down": 32, "channels": 256},
    ]


if __name__ == "__main__":
    model = timm.create_model("maxvit_rmlp_pico_rw_256")
    # model = timm.create_model("maxvit_nano_rw_256")
    feature_extractor = create_feature_extractor(model, ["stem", "stages.0", "stages.1", "stages.2", "stages.3"])
    x = torch.zeros((1, 3, 256, 256))
    features = list(feature_extractor(x).values())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params = {n_params/10**6:.2f} M")
    feature_config = []
    for x in features:
        print(f"{x.shape=}")
        config = {"down": 256 // x.shape[2], "channels": x.shape[1]}
        feature_config.append(config)
    print(f"{feature_config=}")

    model = MaxVitPicoUnet()
    x = torch.zeros((1, 3, 256, 256))
    y = model(x)
    print(f"{y.shape=}")
