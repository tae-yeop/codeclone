from modules import *

class ModelV1(nn.Module):
    """
    [x] UNet Model
    [x] RepBlock
    [x] Multi-branch
    """
    def __init__(self, in_channels, out_channels, 
                 model_channels, num_res_blocks=2, channel_mult=(1,2,4,8), attention_resolutions=[4,2,1], large_kerenl=None, base_kernel=3, freq_domain=False, inference_mode=False
                 ):
        super().__init__()
        self.channel_mult = channel_mult
        self.num_res_blocks= num_res_blocks
        self.inference_mode = inference_mode

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks] # [2,2,2,2]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.base_kernel = base_kernel
        self.large_kernel = 7 if large_kerenl is None else base_kernel
        self.large_kernel_level = int(len(channel_mult)/2)
        
        ch = int(channel_mult[0] * model_channels)
        input_block_chans = [ch]
        stem = [nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)]
        if freq_domain:
            down = WGDown
            up = WGUp
        else:
            down = Downsample
            up = Upsample

        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, ch, kernel_size=3, padding=1,)])
        self.output_blocks = nn.ModuleList([])
        # channel_mult = [1,2,4,4]
        # 0, 1
        # nr = 2
        # 0, 1
        # 
        ds = 1
        for level, mult in enumerate(channel_mult):
            if level <= self.large_kernel_level:
                kernel_size = self.large_kernel
                padding = 3
            else:
                kernel_size = self.base_kernel
                padding = 1
            for nr in range(self.num_res_blocks[level]):
                # print(nr)
                layers = [RepBlock(ch, int(mult * model_channels), kernel_size=kernel_size, padding=padding, inference_mode=inference_mode)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(RepBlock(ch, ch, kernel_size=kernel_size, padding=padding, inference_mode=inference_mode))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            # 0 vs 3
            if level != len(channel_mult) -1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2
            # self.input_blocks.append(nn.Sequential(*layers))

        ######################
        # Bottleneck
        #####################
        self.middle_block = nn.Sequential(
            RepBlock(ch, ch, kernel_size=3, padding=1, inference_mode=inference_mode),
            RepBlock(ch, ch, kernel_size=3, padding=1, inference_mode=inference_mode)
        )

        ######################
        # Decoder
        #####################
        # [(3, 4), (2, 4), (1, 2), (0, 1)]
        for level, mult in list(enumerate(channel_mult))[::-1]:
            if level >= self.large_kernel_level:
                kernel_size = self.large_kernel
                padding = 3
            else:
                kernel_size = self.base_kernel
                padding = 1
            # 0, 1, 2
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [RepBlock(ch + ich, model_channels*mult, kernel_size=kernel_size, padding=padding, inference_mode=inference_mode)]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    layers.append(RepBlock(ch, ch, kernel_size=kernel_size, padding=padding, inference_mode=inference_mode))
                if level and i == self.num_res_blocks[level]:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        self.out = nn.Sequential(RepBlock(model_channels, out_channels, 3, padding=1, zero_params=True, inference_mode=inference_mode))

    def forward(self, x):
        """
        args
        x: [N x C x H x W]
        returns
        an [N x C x H x W]
        """
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        # hs = 12
        # torch.Size([1, 10, 256, 256])
        # torch.Size([1, 10, 256, 256])
        # torch.Size([1, 10, 256, 256])
        # torch.Size([1, 10, 128, 128])
        # torch.Size([1, 20, 128, 128])
        # torch.Size([1, 20, 128, 128])
        # torch.Size([1, 20, 64, 64])
        # torch.Size([1, 40, 64, 64])
        # torch.Size([1, 40, 64, 64])
        #  torch.Size([1, 40, 32, 32])
        # torch.Size([1, 40, 32, 32])
        # torch.Size([1, 40, 32, 32])
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1) # [1, 40, 128, 128] (upsample 때문에) 과 [1, 40, 64, 64]를 concat하려고 함
            h = module(h)
        return self.out(h)
    
# class ModelV2(nn.Module):
#     """
#     [x] UNet Model
#     [x] RepBlock
#     [x] Multi-branch
#     """
#     def __init__(self, in_channels, out_channels, 
#                  model_channels, num_res_blocks=2, channel_mult=(1,2,4,8), attention_resolutions=[4,2,1], large_kerenl=None, base_kernel=3, freq_domain=True
#                  ):
#         super().__init__()
#         self.channel_mult = channel_mult
#         self.num_res_blocks= num_res_blocks

#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks] # [2,2,2,2]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks

#         self.base_kernel = base_kernel
#         self.large_kernel = 7 if large_kerenl is None else base_kernel
#         self.large_kernel_level = int(len(channel_mult)/2)
        
#         ch = int(channel_mult[0] * model_channels)
#         input_block_chans = [ch]
#         stem = [nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)]
#         if freq_domain:
#             down = WGDown
#             up = WGUp
#         else:
#             down = Downsample
#             up = Upsample
#         stem.append(WGDown(use_conv=False))
#         self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, ch, kernel_size=3, padding=1,
#                                                      Downsample())])

#         self.input_blocks.append(stem)
#         self.output_blocks = nn.ModuleList([])

#         # channel_mult = [1,2,4,4]
#         # 0, 1
#         # nr = 2
#         # 0, 1
#         # 
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             if level <= self.large_kernel_level:
#                 kernel_size = self.large_kernel
#                 padding = 3
#             else:
#                 kernel_size = self.base_kernel
#                 padding = 1
#             for nr in range(self.num_res_blocks[level]):
#                 print(nr)
#                 layers = [RepBlock(ch, int(mult * model_channels), kernel_size=kernel_size, padding=padding)]
#                 ch = mult * model_channels
                
#                 if ds in attention_resolutions:
#                     layers.append(RepBlock(ch, ch, kernel_size=kernel_size, padding=padding))
                
#                 self.input_blocks.append(nn.Sequential(*layers))
#                 input_block_chans.append(ch)
#             # 0 vs 3
#             if level != len(channel_mult) -1:
#                 self.input_blocks.append(Downsample(ch, use_conv=True))
#                 input_block_chans.append(ch)
#                 ds *= 2
#             # self.input_blocks.append(nn.Sequential(*layers))
        
#         self.middle_block = nn.Sequential(
#             RepBlock(ch, ch, kernel_size=3, padding=1),
#             RepBlock(ch, ch, kernel_size=3, padding=1)
#         )

#         # [(3, 4), (2, 4), (1, 2), (0, 1)]
#         for level, mult in list(enumerate(channel_mult))[::-1]:
#             if level >= self.large_kernel_level:
#                 kernel_size = self.large_kernel
#                 padding = 3
#             else:
#                 kernel_size = self.base_kernel
#                 padding = 1
#             # 0, 1, 2
#             for i in range(self.num_res_blocks[level] + 1):
#                 ich = input_block_chans.pop()
#                 layers = [RepBlock(ch + ich, model_channels*mult, kernel_size=kernel_size, padding=padding)]
#                 ch = model_channels * mult

#                 if ds in attention_resolutions:
#                     layers.append(RepBlock(ch, ch, kernel_size=kernel_size, padding=padding))
#                 if level and i == self.num_res_blocks[level]:
#                     layers.append(Upsample(ch, use_conv=True))
#                     ds //= 2
#                 self.output_blocks.append(nn.Sequential(*layers))
        
#         self.out = nn.Sequential(RepBlock(model_channels, out_channels, 3, padding=1, zero_params=True))


if __name__ == '__main__':
    model = ModelV1(3, 3, 10, channel_mult=(1,2,4,4))
    print(model(torch.randn(1, 3, 256, 256)).shape)
