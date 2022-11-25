import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from monai.networks.blocks import ResidualUnit, Convolution

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # some layers may not have bias, so skip if this isnt found
        # if hasattr(m, "bias") and m.bias is not None:
        #     nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.BatchNorm3d):
        # nn.init.constant_(m.weight, 1)
        # nn.init.constant_(m.bias, 0)

class MultiConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size=3,
    ) -> None:
        super(MultiConv, self).__init__()

        self.dimensions = 3
        self.num = len(channels)
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size=kernel_size


        conv_tmp = []
        for i in range(self.num):
            conv_tmp.append(
                Convolution(
                    self.dimensions,
                    in_channels = in_channels if i==0 else channels[i-1],
                    out_channels=self.channels[i],
                    strides=self.strides[i],
                    kernel_size=self.kernel_size,
                    act='PRELU',
                    norm="batch",
                    dropout=0.0,
                    bias=True,
                    conv_only=False,
                    is_transposed=False,
                    adn_ordering='NDA',
                )
            )

        self.convs = nn.Sequential(*conv_tmp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)

class PredDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int
    ) -> None:
        super(PredDecoder, self).__init__()

        self.dimensions = 3
        self.in_channels = in_channels

        self.pred_conv = nn.Sequential(
            Convolution(
                self.dimensions,
                in_channels=in_channels,
                out_channels=in_channels//2,
                strides=2,
                kernel_size=3,
                act='PRELU',
                norm='batch',
                dropout=0.0,
                bias=False,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            Convolution(
                self.dimensions,
                in_channels=in_channels//2,
                out_channels=in_channels // 4,
                strides=2,
                kernel_size=3,
                act='PRELU',
                norm='batch',
                dropout=0.0,
                bias=False,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            Convolution(
                self.dimensions,
                in_channels=in_channels//4,
                out_channels=2,
                strides=1,
                kernel_size=3,
                padding=1,
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=False,
            )
        )

    def forward(self, x):
        return self.pred_conv(x)


class My_Unet_3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        norm="batch"
    ) -> None:
        super(My_Unet_3D, self).__init__()

        self.dimensions = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels  # [16, 32, 64, 128, 256]
        self.strides = strides  # [2, 2, 2, 2]
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.norm = norm

        # downsampling
        self.down0 = ResidualUnit(
            spatial_dims=self.dimensions,
            in_channels=self.in_channels,
            out_channels=16,
            strides=2,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act='PRELU',
            norm=self.norm,
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )
        self.down1 = ResidualUnit(
            spatial_dims=self.dimensions,
            in_channels=16,
            out_channels=32,
            strides=2,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act='PRELU',
            norm=self.norm,
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )
        self.down2 = ResidualUnit(
            spatial_dims=self.dimensions,
            in_channels=32,
            out_channels=64,
            strides=2,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act='PRELU',
            norm=self.norm,
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )
        self.down3 = ResidualUnit(
            spatial_dims=self.dimensions,
            in_channels=64,
            out_channels=128,
            strides=2,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act='PRELU',
            norm=self.norm,
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )
        self.down4 = ResidualUnit(
            spatial_dims=self.dimensions,
            in_channels=128,
            out_channels=256,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act='PRELU',
            norm=self.norm,
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )

        # upsampling
        self.up0 = nn.Sequential(
            Convolution(
                self.dimensions,
                in_channels=32,
                out_channels=2,
                strides=2,
                kernel_size=self.up_kernel_size,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=2,
                out_channels=2,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                last_conv_only=True,  # 最后一层
                adn_ordering='NDA',
            )
        )
        self.up1 = nn.Sequential(
            Convolution(
                self.dimensions,
                in_channels=64,
                out_channels=16,
                strides=2,
                kernel_size=self.up_kernel_size,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=16,
                out_channels=16,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                adn_ordering='NDA',
            )
        )
        self.up2 = nn.Sequential(
            Convolution(
                self.dimensions,
                in_channels=128,
                out_channels=32,
                strides=2,
                kernel_size=self.up_kernel_size,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=32,
                out_channels=32,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                adn_ordering='NDA',
            )
        )
        self.up3 = nn.Sequential(
            Convolution(
                self.dimensions,
                in_channels=512,
                out_channels=64,
                strides=2,
                kernel_size=self.up_kernel_size,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                conv_only=False,
                is_transposed=True,
                adn_ordering='NDA',
            ),
            ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=64,
                out_channels=64,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act='PRELU',
                norm=self.norm,
                dropout=0.0,
                bias=True,
                adn_ordering='NDA',
            )
        )

        # # initialize weights
        # for m in self.modules():
        #     init_weights(m)

        self.first_conv = MultiConv(
            in_channels=33,
            channels=[64, 64, 64, 128, 128, 128],
            strides=[2, 1, 1, 2, 1, 1],
        )
        self.first_pred_decoder = PredDecoder(
            in_channels=128
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs                                                                    (1,4,128,128,48)
        [img, pos_mask_geodis_first, pos_mask_gauss, neg_mask_gauss] = torch.split(inputs,1,dim=1)
        img_with_anno = torch.cat((img, pos_mask_gauss, neg_mask_gauss), dim=1)   # (1,3,128,128,48)
        down0 = self.down0(img_with_anno)           # 3->16                         (1,16,64,64,24)
        down1 = self.down1(down0)                   # 16->32                        (1,32,32,32,12)
        down1_first = torch.cat((down1,
                                 F.interpolate(                                   # (1,1,32,32,12)
                                     pos_mask_geodis_first,
                                     size=down1.size()[2:],
                                     mode='trilinear',
                                     align_corners=True
                                 )), dim=1)                                       # (1,33,32,32,12)
        down1_first = self.first_conv(down1_first)                                # (1,128,8,8,3)
        result_first = self.first_pred_decoder(down1_first)                       # (1,1,32,32,12)
        result_first = F.interpolate(
            result_first,
            size=img.size()[2:],
            mode='trilinear',
            align_corners=True                                                    # (1,1,128,128,48)
        )
        down2 = self.down2(down1)                   # 32->64                        (1,64,16,16,6)
        down3 = self.down3(down2)                   # 64->128                       (1,128,8,8,3)
        down4 = self.down4(down3)                   # 128->256                      (1,256,8,8,3)
        up4 = torch.cat([down3, down4, down1_first], dim=1)  # 128+256+128=512   (1,512,8,8,3)
        up3 = self.up3(up4)  # 512->64                              (1,64,16,16,6)
        up3 = torch.cat([down2, up3], dim=1)  # 64+64=128           (1,128,16,16,6)
        up2 = self.up2(up3)  # 128->32                              (1,32,32,32,12)
        up2 = torch.cat([down1, up2], dim=1)  # 32+32=64            (1,64,32,32,12)
        up1 = self.up1(up2)  # 64->16                               (1,16,64,64,24)
        up1 = torch.cat([down0, up1], dim=1)  # 16+16=32            (1,32,64,64,24)
        up0 = self.up0(up1)  # 32->2                                (1,2,128,128,48)

        return [up0, result_first]                                  # (1,3,128,128,48)