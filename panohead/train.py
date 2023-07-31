import inspect
import logging

import datasets
from datasets import load_dataset
import accelerator

import torchvision.transforms as transforms


def main(args):
    # logging

    # seed
    if args.seed is not None:
        set_seed(args.seed)

        
    # 모델 초기화
    # config = PanoHead.load_config(args.model_config_name_or_path)
    # model = PanoHead.from_config(config)

    generator = 



    # datasets을 활용
    # 미리 row, column을 만들어 두도록 한다
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
                               cache_dir=args.cache_dir,
                               split="train",)

    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")


    augmentations = ...

    train_transforms = transforms.Compose([
        transforms.Resize(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 메인 프로세스에ㅓ preprocessing을 수행 후 
    # 다른 프로세스에도 전달한다
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = 

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )