import os
from functools import partial
import torch
from torchvision.datasets import Kinetics400
from torchvision.transforms import transforms
from PIL import Image
import numpy as np


def trunc_normal_(x, mean=0., std=1.):
  "Truncated normal initialization (approximation)"
  # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
  return x.normal_().fmod_(2).mul_(std).add_(mean)


def vid_transform_fn(x, fn):
  return [fn(Image.fromarray(X.squeeze(dim=0).data.numpy())) for X in x]


def video_collate(batch):
  is_np = isinstance(batch[0][0][0], np.ndarray)
  T = len(batch[0][0])  # number of frames
  targets = torch.tensor([b[2] for b in batch])
  if len(batch[0]) == 3:
    extra_data = [b[1] for b in batch]
  else:
    extra_data = []
  batch_size = len(batch)
  if is_np:
    dims = (batch[0][0][0].shape[2], batch[0][0][0].shape[0], batch[0][0][0].shape[1])
    tensor_uint8_CHW = torch.empty((T * batch_size, *dims), dtype=torch.uint8)
    for i in range(batch_size):
      for t in range(T):
        tensor_uint8_CHW[i * T + t] = \
          torch.from_numpy(batch[i][0][t]).permute(2, 0, 1)
    return tensor_uint8_CHW, targets

  else:
    dims = (batch[0][0][0].shape[0], batch[0][0][0].shape[1], batch[0][0][0].shape[2])
    tensor_float_CHW = torch.empty((T * batch_size, *dims), dtype=torch.float)
    for i in range(batch_size):
      for t in range(T):
        tensor_float_CHW[i * T + t] = batch[i][0][t]
    return tensor_float_CHW, targets


def create_val_dataset(args, transform, add_extra_data=True):
  source = args.val_dir

  valid_data = Kinetics400(root=source,
                                step_between_clips=args.step_between_clips,
                                frames_per_clip=args.frames_per_clip, frame_rate=args.frame_rate,
                                extensions=('avi', 'mp4'),
                                transform=partial(vid_transform_fn, fn=transform))
  return valid_data


def create_dataloader(args):
  val_bs = args.batch_size
  if args.input_size == 448:  # squish
    val_tfms = transforms.Compose(
      [transforms.Resize((args.input_size, args.input_size))])
  else:  # crop
    val_tfms = transforms.Compose(
      [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
       transforms.CenterCrop(args.input_size)])
  val_tfms.transforms.append(transforms.ToTensor())
  val_dataset = create_val_dataset(args, val_tfms)
  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=val_bs, shuffle=False,
    num_workers=args.num_workers, collate_fn=video_collate, pin_memory=True, drop_last=False)
  return val_loader


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self): self.reset()

  def reset(self): self.val = self.avg = self.sum = self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def validate(model, val_loader):
  prec1_m = AverageMeter()
  prec5_m = AverageMeter()
  last_idx = len(val_loader) - 1

  with torch.no_grad():
    for batch_idx, (input, target) in enumerate(val_loader):
      last_batch = batch_idx == last_idx
      input = input.cuda()
      target = target.cuda()
      output = model(input)

      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      prec1_m.update(prec1.item(), output.size(0))
      prec5_m.update(prec5.item(), output.size(0))

      if (last_batch or batch_idx % 100 == 0):
        log_name = 'Kinetics Test'
        print(
          '{0}: [{1:>4d}/{2}]  '
          'Prec@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '
          'Prec@5: {top5.val:>7.2f} ({top5.avg:>7.2f})'.format(
            log_name, batch_idx, last_idx,
            top1=prec1_m, top5=prec5_m))
  return prec1_m, prec5_m

