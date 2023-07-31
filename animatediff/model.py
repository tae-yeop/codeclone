import torch

a = [torch.randn(3, 10)]*5
b = torch.concat(a)
# print(b.shape)


import inspect

def foo(a, b, c=3, *args, **kwargs):
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)
    print(values)

foo(1, 2, 3, 4, 5, x=6, y=7)


t = torch.tensor(1)
t = t[None]
t = t.expand(10)
print(len(t.shape), t.shape)

import torch

a = [torch.randn(3, 10)]*5
b = torch.concat(a)
# print(b.shape)


import inspect

def foo(a, b, c=3, *args, **kwargs):
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)
    print(values)

foo(1, 2, 3, 4, 5, x=6, y=7)


t = torch.tensor(1)
t = t[None]
t = t.expand(10)
print(len(t.shape), t.shape)


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,

    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DonwBlock3D":
        return DownBlock3D()

    elif donw_block_type == "CrossAttnDownBlock3D":
        return CrossAttnDownBlock3D()
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,

):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D()

    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_Dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnDownBlock3D()


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(self,
                 in_channels,
                 
    ):
        ...
        resnets = [
            ResnetBlock3D()
        ]
        attentions = []
        motion_modules = []
        
        for _ in range(num_layers):
            attentions.append(Transformer3DModel(attn_num_head_channels,
                                                 in_channels // attn)) # num_attention_heads

            motion_modules.append(get_motion_module
                                  (in_channels = in_channels,
                                   motion_module_type=motion_module_type,
                                   motion_module_kwargs=motion_module_kwargs) if use_motion_module else None)

            resnets.append(ResnetBlock3D())
            
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        hidden_states = self.resnet[0](hidden_states, temb)
        for attn, resnet, motion_module in zip(self,attentions, self.resnet[1:], self.motion_modules):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states
            hidden_states = resnet(hidden_states, temb)

        return hidden_states



class CrossAttnDownBlock3D(nn.Module):
    def __init__(self,
                 unet_use_cross_frame_attention=None,
                 unet_use_temporal_attention=None,
                 use_motion_module=None,
                 motion_module_type=None,
                 motion_module_kwargs=None):
        ...

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample3D(
                out_channels, use_conv
            )])

    def forward(self, hidden_states, tmeb=None, encoder_hidden_states=None, attention_mask=None):
        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*input):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward


                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet))
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(motion_module), hidden_states.requires_grad_(), temb, encoder_hidden_states)

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).samples
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

            output_states += (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ...,
                 use_motion_module=None,
                 motion_module_type=None,
                 motion_module_kwargs=None):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=tmeb_channels,
                    eps=resnet_eps,
                    ...
                )
            )
            motion_modules.append(get_motion_module(
                in_channels=out_channels,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            ) if use_motion_module else None)


        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [DownBlock3D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # checkpoint를 사용한다면
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, tmeb)
                if motion_module is not None:

            else:
                hidden_states = resnet(hidden_states, temb)

                hidden_states = motion_module(hidden_states,temb,encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, out
        