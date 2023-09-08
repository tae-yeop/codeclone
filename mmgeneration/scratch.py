class DynamicIterBasedRunner(IterBasedRunner):
    def __init__(self,
                    *args,
                    is_dynamic_ddp=False,
                    pass_training_status=False,
                    fp16_loss_scaler=None,
                    use_apex_amp=False,
                    **kwargs):
        
        """
        data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                    and validation.
        workflow (list[tuple]): A list of (phase, iters) to specify the
            running order and iterations. E.g, [('train', 10000),
            ('val', 1000)] means running 10000 iterations for training and
            1000 iterations for validation, iteratively.
        """
        super().__init__(*args, **kwargs)
        ...
        self.is_dynamic_ddp = is_dynamic_ddp

        

    def run(self, data_loader, workflow, max_iters=None, **kwargs):
        iter_loaders = [IterLoader(x, self) for x in data_loaders]
        
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0 
                mode, iters = flow
                ...
                iter_runner = getattr(self, mode) # 여기서 self.train에 대해 가리킴
                for _ in range(iters):
                    if mode =='train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)
        ...

    def train(sef, data_loader, **kwargs):
        if is_module_warpper(self.model):
            _model = self.model.module
        else:
            _model=  self.model

        self.model.train()
        self.mode = 'train'

        ...
        self.data_loader = data_loader
        data_batch = next(self.data_loader)

        # ddp reducer for tracking dynamic computational graph
        if self.is_dynamic_ddp:
            kwargs.update(dict(ddp_reducer=self.model.reducer))

        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

        ...



# mmgen/core/ddp_wrapper.py 
# 개별적인 모델에 적용
from mmcv.parallel import MODULE_WRAPPERS, MMDistributedDataParallel

@MODULE_WRAPPERS.register_module('mmgen.DDPWrapper')
class DistributedDataParallelWrapper(nn.Module):
    def __init__(self, module, device_ids, dim=0, broadcast_buffers=False,
                 find_unused_parameters=False,
                 **kwargs):

        super().__init__()

        self.to_ddp(device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)

    def to_ddp(self, device_ids, dim, broadcast_buffers,
               find_unused_parameters, **kwargs):
        for name, module in self.moudle._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)

            self.module._modules[name] = module

    def train_step(self, *inputs, **kwargs):
        """Train step function.
        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        # 인풋과 kwargs를 먼저 scatter
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        # MMDDP Wrapper 실행
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output


# mmcv/parallel/distributed.py 
# torch.nn.parallel.distributed.DistributedDataParallel를 상속함
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)
class MMDistributedDataParallel(DistributedDataParallel):

    def train_step(self, *inputs, **kwargs):
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.

        # Forward 수행하는 부분
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # self.module은 DDP 이전의 원본 모듈
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        #  Forward 이후에 buffer snyc
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()
                
        # grad engine 켜진 상태이고 backward에서 sync 맞춰야 하는 경우
        if (torch.is_grad_enabled() and getattr(self, 'require_backward_grad_sync', False)
            and self.require_backward_grad_sync):
            # 안쓰는 parameter 있는 경우
            if self.find_unused_parameters:
                # backwrad를 위한 reducer를 다시 준비한다.
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        # 그냥 no_grad 상태이면 무시
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
                
        return output




        