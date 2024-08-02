import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epochs between validation phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='dataset/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Threshold for filtering predictions during validation')

    args = parser.parse_args()
    return args


def postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold,
                mask=None):
    classification = torch.sigmoid(classification)
    regression = torch.sigmoid(regression)

    if mask is not None:
        classification = classification * mask

    preds = []
    for i in range(len(imgs)):
        pred_boxes = []
        # Example: Replace this with actual bbox processing logic
        # Apply threshold to classification scores
        scores = classification[i].max(dim=1)[0]
        keep = scores > threshold

        # Implement NMS if needed here

        pred_boxes.append({
            'boxes': [],  # Fill with processed boxes
            'scores': scores[keep]  # Filtered scores
        })

    return preds


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def collate_fn(batch):
    imgs = [item['img'] for item in batch]
    annot = [item['annot'] for item in batch]
    filenames = [item['filename'] for item in batch]
    scales = [item['scale'] for item in batch]

    imgs = torch.stack(imgs, 0)
    annot = torch.stack(annot, 0)

    return {'img': imgs, 'annot': annot, 'filename': filenames, 'scale': scales}


def train(opt):
    params = Params(f'project/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    input_sizes = [512, 640, 768, 896, 1024]  # Example sizes for different coefficients

    train_dataset = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                                transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                              Augmenter(),
                                                              Resizer(input_sizes[opt.compound_coef])]))

    val_dataset = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                              transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef])]))

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collate_fn,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collate_fn,
                  'num_workers': opt.num_workers}

    training_generator = DataLoader(train_dataset, **training_params)
    val_generator = DataLoader(val_dataset, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Optimizer must be adamw or sgd')

    for epoch in range(last_step // len(training_generator) + 1, opt.num_epochs + 1):
        model.train()
        for step, (imgs, annotations, _) in tqdm(enumerate(training_generator)):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                annotations = [ann.cuda() for ann in annotations]

            optimizer.zero_grad()
            cls_loss, reg_loss = model(imgs, annotations)
            loss = cls_loss + reg_loss
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(training_generator) + step)
                writer.add_scalar('Train/Cls_Loss', cls_loss.item(), epoch * len(training_generator) + step)
                writer.add_scalar('Train/Reg_Loss', reg_loss.item(), epoch * len(training_generator) + step)

        if epoch % opt.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for imgs, annotations, _ in tqdm(val_generator):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        annotations = [ann.cuda() for ann in annotations]
                    cls_loss, reg_loss = model(imgs, annotations)
                    loss = cls_loss + reg_loss
                    val_loss += loss.item()

                writer.add_scalar('Val/Loss', val_loss / len(val_generator), epoch)

        if epoch % opt.save_interval == 0:
            save_path = os.path.join(opt.saved_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    writer.close()


if __name__ == "__main__":
    args = get_args()
    train(args)