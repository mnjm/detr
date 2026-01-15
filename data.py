"""
Dataset and dataloader utilities.

- Currently only supports VOC 2007 + 2012 datasets downloaded using the convient `download-voc-dataset.sh` bash script
- Defines a PyTorch Dataset to load these datasets.
- Builds train/val dataloaders with torchvision v2 transforms, augmentations are controlled through hydra config.
- There is also a batch visualizer for debugging.
"""

import torch
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import tv_tensors
from torchvision.io import read_image
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from utils import to_2tuple
from bidict import bidict

supported_datasets = ["voc_2007_2012"]

class VOCObjectDataset(Dataset):
    """
    PyTorch Dataset for Pascal VOC 2007/2012 object detection.

    Each sample returns:
        image: Tensor[C, H, W]
        targets: dict with
            - 'bboxes': tv_tensors.BoundingBoxes (XYXY)
            - 'labels': Tensor[N]
    """

    def __init__(
            self, root: Path, split: str, class_map: bidict, transforms: T.Compose = None,
            max_objs: int=32
        ):
        """
        Args:
            root (Path): Root directory containing dataset splits.
            split (str): One of ['trainval', 'test'].
            class_map (bidict): idx to class_map bidict.
            transforms (T.Compose): torchvision v2 transforms applied jointly on image & targets.
            max_objs (int): Maximum allowed objects per image.
        """
        self.transforms = transforms
        self.class_map = class_map
        self.max_objs = max_objs

        img_set_fp = root / "ImageSets" / "Main" / f"{split}.txt"
        assert img_set_fp.is_file(), f"{img_set_fp!r} not found."

        with img_set_fp.open('r') as f:
            self.img_id_l = [ line.strip() for line in f.readlines() ]

        self.ann_dir = root / "Annotations"
        assert self.ann_dir.is_dir(), f"{self.ann_dir!r} not found."
        self.jpg_dir = root / "JPEGImages"
        assert self.jpg_dir.is_dir(), f"{self.jpg_dir!r} not found."

    @staticmethod
    def _parse_xml(xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root.findall('object')

    def __len__(self):
        return len(self.img_id_l)

    def __getitem__(self, idx: int):
        img_id = self.img_id_l[idx]
        obj_l = VOCObjectDataset._parse_xml(self.ann_dir / f"{img_id}.xml")
        img_path = self.jpg_dir / f"{img_id}.jpg"

        img = read_image(str(img_path))

        bbox_l = torch.zeros((len(obj_l), 4), dtype=torch.float32)
        lbl_l = torch.zeros((len(obj_l),), dtype=torch.int64)
        area_l = torch.zeros((len(obj_l),), dtype=torch.float32)
        for idx, obj in enumerate(obj_l):
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x1, y1, x2, y2 = [
                float(bbox.find(tag).text)-1 for tag in ["xmin", "ymin", "xmax", "ymax"]
            ] # 1 indexed to 0 indexed
            cls_idx = self.class_map.inv[cls_name]

            lbl_l[idx] = cls_idx
            bbox_l[idx, ...] = torch.as_tensor([x1, y1, x2, y2], dtype=torch.float32)
            area_l[idx] = (x2 - x1) * (y2 - y1)

        if len(obj_l) > self.max_objs:
            idxs = torch.argsort(area_l, descending=True)[:self.max_objs]
            bbox_l = bbox_l[idxs]
            lbl_l = lbl_l[idxs]

        targets = {
            'bboxes': tv_tensors.BoundingBoxes(bbox_l, format='xyxy', canvas_size=img.shape[1:]),
            'labels': lbl_l,
        }

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        h, w = img.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor
        return img, targets

def collate_fn(batch):
    """
    Custom collate function for dataloader, Stacks images but keeps targets as a list as no. of objects varies per image.
    """
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)

def init_class_map(dataset_name: str, background_idx: int = 0):
    assert dataset_name in supported_datasets
    class_map = {}
    if dataset_name == "voc_2007_2012":
        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]
        voc_classes.insert(background_idx, "background")
        class_map = { i: cls for i, cls in enumerate(voc_classes) }
        class_map = bidict(class_map)
    return class_map

def init_dataloaders(cfg: object, split: str):
    """
    Creates dataset and dataloader for a given split.

    - Downloads Pascal VOC 2012 (COCO format) if missing
    - Builds torchvision v2 transforms via Hydra config

    Args:
        cfg: Hydra config object
        split (str): 'train' or 'val'

    Returns:
        DataLoader
    """

    dataset_name = cfg.dataset_name
    assert dataset_name in supported_datasets, f"{cfg.dataset_name!r} not supported"
    assert split in ['train', 'val'], f"{split=} not valid"

    transforms = []
    if split == 'train':
        for aug in cfg.augmentations:
            transforms.append(instantiate(aug, _convert_="all"))
    transforms.append(T.Resize(to_2tuple(cfg.input_size)))
    if split == "train":
        transforms.append(T.SanitizeBoundingBoxes(labels_getter=lambda inp: (inp[1]["labels"])))
    transforms.extend([
        T.ToPureTensor(),
        T.ToDtype(dtype=torch.float32, scale=True),
        T.Normalize(mean=cfg.input_normalization.mean, std=cfg.input_normalization.std)
    ])
    transforms = T.Compose(transforms)

    class_map = init_class_map(dataset_name, cfg.model.background_cls_idx)
    max_objs = cfg.model.n_queries
    if dataset_name == "voc_2007_2012":
        common_path = Path("./dataset/voc-datasets")
        if split == "train":
            dataset_l = [ VOCObjectDataset(
                common_path / folder_name , split="trainval", transforms=transforms,
                max_objs=max_objs, class_map=class_map
            ) for folder_name in ['VOC2007', 'VOC2012'] ]
            dataset = ConcatDataset(dataset_l)
        elif split == "val":
            dataset = VOCObjectDataset(
                common_path / "VOC2007", split="test", transforms=transforms,
                max_objs=max_objs, class_map=class_map
            )

    dataloader_kwargs = dict(cfg.dataloader)
    dataloader_kwargs['drop_last'] = split=="train" and dataloader_kwargs.get('drop_last', False)
    dataloader_kwargs['shuffle'] = split=="train"
    dataloader = DataLoader(dataset, **dataloader_kwargs, collate_fn=collate_fn)
    dataloader.class_map = class_map

    return dataloader

def show_batch(dataloader: DataLoader, N: int, nrow: int=None):
    batch = next(iter(dataloader))
    imgs, targets = batch
    B = imgs.shape[0]
    assert B >= N, f"{N=} should be <= batch size {B}"
    imgs = imgs[:N].detach().cpu().permute(0, 2, 3, 1).numpy()
    # normalize image
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)

    nrow = nrow if nrow else math.ceil(math.sqrt(N))
    ncol = math.ceil(N / nrow)
    figsize_scale = 2.0
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * figsize_scale, ncol * figsize_scale))
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= N:
            break
        ax.imshow(imgs[i])
        h, w = imgs[i].shape[:2]
        bbox_scale = torch.tensor([w, h, w, h], device="cpu")

        bboxes, lbl_idx_l = targets[i]['bboxes'].cpu(), targets[i]['labels'].cpu()
        for bbox, lbl_idx in zip(bboxes, lbl_idx_l):
            bbox = bbox * bbox_scale
            x1, y1, x2, y2 = bbox.tolist()
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=1.5)
            ax.add_patch(rect)
            lbl_name = dataloader.class_map[lbl_idx.item()]
            ax.text(
                x1, y1 - 2, lbl_name, color="red", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )

    plt.tight_layout()
    plt.show()
    plt.close(fig)