import os
import csv
import yaml
import torch
import argparse
import datetime
import traceback
import numpy as np
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from efficientdet.loss import FocalLoss
from backbone import EfficientDetBackbone
from utils.sync_batchnorm import patch_replication_callback
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string

# Define CSV logging
log_file_path = 'training_logs.csv'

def initialize_csv_log():
    with open(log_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Step', 'Epoch', 'Iteration', 'Cls Loss', 'Reg Loss', 'Total Loss',
            'Val Epoch', 'Cls Loss Val', 'Reg Loss Val', 'Total Loss Val'
        ])

def append_csv_log(step, epoch, iter, cls_loss, reg_loss, total_loss, val_epoch=None, cls_loss_val=None, reg_loss_val=None, total_loss_val=None):
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            step, epoch, iter, cls_loss, reg_loss, total_loss,
            val_epoch, cls_loss_val, reg_loss_val, total_loss_val
        ])

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Diabetic Foot')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
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

    args = parser.parse_args()
    return args

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

def train(opt):
    params = Params(f'project/{opt.project}.yml')
    initialize_csv_log()  # Initialize the CSV log file

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

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
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
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
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
            patch_replication_callback(model)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=4e-5)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    else:
        raise NotImplementedError(f'{opt.optim} not implemented')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.3)

    step = last_step

    while step < len(training_generator) * opt.num_epochs:
        model.train()
        epoch = step // len(training_generator)
        epoch_iter = step % len(training_generator)
        pbar = tqdm(enumerate(training_generator), total=len(training_generator), leave=False)

        for i, (imgs, annotations, obj_list) in pbar:
            if epoch == opt.num_epochs:
                break

            if params.num_gpus > 0:
                imgs = imgs.cuda()
                annotations = [a.cuda() for a in annotations]

            cls_loss, reg_loss = model(imgs, annotations, obj_list)
            total_loss = cls_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_description(f'[Training] Epoch: {epoch}, Iter: {i}, Cls Loss: {cls_loss.item():.4f}, '
                                 f'Reg Loss: {reg_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')
            writer.add_scalar('training/cls_loss', cls_loss.item(), step)
            writer.add_scalar('training/reg_loss', reg_loss.item(), step)
            writer.add_scalar('training/total_loss', total_loss.item(), step)

            # Log details to CSV
            append_csv_log(step, epoch, i, cls_loss.item(), reg_loss.item(), total_loss.item())

            step += 1

        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_generator), total=len(val_generator), leave=False)
            cls_loss_val, reg_loss_val, total_loss_val = 0, 0, 0
            for i, (imgs, annotations, obj_list) in val_pbar:
                if params.num_gpus > 0:
                    imgs = imgs.cuda()
                    annotations = [a.cuda() for a in annotations]

                cls_loss, reg_loss = model(imgs, annotations, obj_list)
                total_loss = cls_loss + reg_loss
                cls_loss_val += cls_loss.item()
                reg_loss_val += reg_loss.item()
                total_loss_val += total_loss.item()

                val_pbar.set_description(f'[Validation] Epoch: {epoch}, Cls Loss: {cls_loss.item():.4f}, '
                                         f'Reg Loss: {reg_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')
                writer.add_scalar('validation/cls_loss', cls_loss.item(), step)
                writer.add_scalar('validation/reg_loss', reg_loss.item(), step)
                writer.add_scalar('validation/total_loss', total_loss.item(), step)

            # Log validation details to CSV
            append_csv_log(step, epoch, i, cls_loss.item(), reg_loss.item(), total_loss.item(), epoch, cls_loss_val, reg_loss_val, total_loss_val)

        scheduler.step()

        if (epoch + 1) % opt.val_interval == 0:
            print(f'[Info] Saving weights at epoch {epoch + 1}...')
            torch.save(model.state_dict(), opt.saved_path + f'{epoch + 1}.pth')

        if epoch >= opt.num_epochs - 1:
            print(f'[Info] Training completed.')
            break

if __name__ == '__main__':
    args = get_args()
    try:
        train(args)
    except Exception as e:
        print(f'[Error] {e}')
        traceback.print_exc()