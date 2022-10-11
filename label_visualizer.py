import argparse
import glob
import os
from collections import namedtuple
from typing import Tuple

import cv2
import torch
import torchvision
import tqdm
from torch import Tensor

CelebAMaskHQClass = namedtuple('CelebAMaskHQClass', ['name', 'id', 'color'])
classes = [
    CelebAMaskHQClass('background', 0, (17, 17, 17)),
    CelebAMaskHQClass('skin', 1, (255, 178, 102)),
    CelebAMaskHQClass('l_brow', 2, (0, 255, 128)),
    CelebAMaskHQClass('r_brow', 3, (153, 51, 255)),
    CelebAMaskHQClass('l_eye', 4, (51, 255, 255)),
    CelebAMaskHQClass('r_eye', 5, (102, 102, 255)),
    CelebAMaskHQClass('eye_g', 6, (255, 0, 0)),
    CelebAMaskHQClass('l_ear', 7, (122, 49, 0)),
    CelebAMaskHQClass('r_ear', 8, (1, 72, 79)),
    CelebAMaskHQClass('ear_r', 9, (244, 225, 255)),
    CelebAMaskHQClass('nose', 10, (72, 0, 0)),
    CelebAMaskHQClass('mouth', 11, (255, 255, 204)),
    CelebAMaskHQClass('u_lip', 12, (0, 0, 204)),
    CelebAMaskHQClass('l_lip', 13, (255, 80, 80)),
    CelebAMaskHQClass('neck', 14, (102, 255, 102)),
    CelebAMaskHQClass('neck_l', 15, (36, 3, 235)),
    CelebAMaskHQClass('cloth', 16, (255, 0, 127)),
    CelebAMaskHQClass('hair', 17, (204, 255, 229)),
    CelebAMaskHQClass('hat', 18, (255, 255, 0)),
]
colors = [cls.color for cls in classes]


def draw_segmentation_mask(image: Tensor, mask: Tensor, colors: list, alpha=0.4, gamma=20) -> Tuple[Tensor, Tensor]:
    assert image.dtype == torch.uint8, f'The images dtype must be uint8, got {image.dtype}'
    assert image.dim() == 3, 'The images must be of shape (C, H, W)'
    assert image.size()[0] == 3, 'Pass RGB images. Other Image formats are not supported'
    assert image.shape[-2:] == mask.shape[-2:], 'The images and the masks must have the same height and width'
    assert mask.ndim == 2, 'The masks must be of shape (H, W)'
    assert mask.dtype == torch.uint8, f'The masks must be of dtype uint8. Got {mask.dtype}'
    assert image.device == mask.device, 'The device of images and masks must be the same'
    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1. 0 means full transparency, 1 means no transparency'
    assert len(colors[0]) == 3, 'The colors must be RGB format'

    h, w = mask.size()
    colored_mask = torch.zeros((3, h, w), dtype=torch.uint8, device=mask.device)
    r = colored_mask[0, :, :]
    g = colored_mask[1, :, :]
    b = colored_mask[2, :, :]
    for i, color in enumerate(colors):
        r[mask == i] = color[0]
        g[mask == i] = color[1]
        b[mask == i] = color[2]

    alpha_colored_mask = image * (1 - alpha) + colored_mask * alpha + gamma
    alpha_colored_mask = alpha_colored_mask.clamp(0, 255).to(torch.uint8)
    return alpha_colored_mask, colored_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='0031', help='Directory where source data is stored')
    parser.add_argument('--dst', type=str, default='results', help='directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='device to use (auto, cpu, cuda)')
    args = parser.parse_args()
    print(args)

    # Device 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # source 폴더에서 데이터 경로 로드
    assert os.path.exists(args.src), 'src is not exists'
    assert os.path.isdir(args.src), 'src must be a directory'
    labels = glob.glob(os.path.join(args.src, '*_grayscale.png'))
    images = [i.replace('_grayscale.png', '.png') for i in labels]
    assert len(images) > 0 and len(labels) > 0, 'No file in src'
    assert len(images) == len(labels)

    # 정렬
    try:
        images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        labels.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        images.sort()
        labels.sort()

    # 라벨의 클래스 구성 정보 저장
    label_class_ids = torch.zeros(0, dtype=torch.int32, device=device)

    color_mixed_label_dir = os.path.join(args.dst, 'mix')
    colored_label_dir = os.path.join(args.dst, 'color')
    os.makedirs(color_mixed_label_dir, exist_ok=True)
    os.makedirs(colored_label_dir, exist_ok=True)
    for image_path, label_path in tqdm.tqdm(zip(images, labels), 'Process', total=len(images)):
        assert os.path.exists(image_path), f'source image is not exists: {image_path}'
        assert os.path.exists(label_path), f'label image is not exists: {label_path}'

        # 이미지 로드
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = torch.as_tensor(image, device=device).permute((2, 0, 1))
        label = torch.as_tensor(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE), device=device)

        # 라벨의 클래스 구성 정보 모음
        label_class_ids = torch.cat((label_class_ids, label.unique()))

        # 라벨 데이터로 색칠 (혼합 라벨, 색칠 라벨)
        color_mixed_label, colored_label = draw_segmentation_mask(image, label, colors)

        # 색칠한 결과 저장
        file_name = os.path.basename(label_path)
        torchvision.io.write_png(color_mixed_label.cpu(), os.path.join(color_mixed_label_dir, file_name))
        torchvision.io.write_png(colored_label.cpu(), os.path.join(colored_label_dir, file_name))

    label_class_ids = label_class_ids.unique().tolist()
    print(f'라벨의 클래스 구성 정보({len(label_class_ids)}개): {label_class_ids}')
