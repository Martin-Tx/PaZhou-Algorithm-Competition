# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral

import traceback
import uuid
import random
import math
import numpy as np
import os
import copy
import logging
import cv2
from PIL import Image, ImageDraw
import pickle
from paddle.fluid.dataloader.collate import default_collate_fn

from modeling import bbox_utils
from paddleseg.transforms import functional
from ppdet.data.transform import RandomResize, RandomResizeCrop, Pad
import logging
logger = logging.getLogger(__name__)
import time 
import imgaug.augmenters as iaa
from .op_helper import (satisfy_sample_constraint, filter_and_process,
                        generate_sample_bbox, clip_bbox, data_anchor_sampling,
                        satisfy_sample_constraint_coverage, crop_image_sampling,
                        generate_sample_bbox_square, bbox_area_sampling,
                        is_poly, get_border)

class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                # op_cls = getattr(detection_ops, k)
                op_cls = eval(k)
                # print('v:', v)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e
        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        start = time.time()
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e
        # print(f'dections data_time: {time.time() - start} s')
        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


class ImageError(ValueError):
    pass


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


class Decode(BaseOperator):
    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        im = im.transpose((2, 0, 1))
        sample['image'] = im
        return sample

class IAANoise(BaseOperator):
    def __init__(self, prob=0.5):
        super(IAANoise, self).__init__()

        self.iaanoise = []
        scale = np.random.randint(0, 500) / 10000.
        self.iaanoise.append(iaa.AdditiveGaussianNoise(scale=scale*255, per_channel=True))
        self.iaanoise.append(iaa.imgcorruptlike.ShotNoise(severity=[1,1,1]))
        self.iaanoise.append(iaa.imgcorruptlike.ImpulseNoise(severity=[1,1,1]))
        self.iaanoise.append(iaa.imgcorruptlike.SpeckleNoise(severity=[1,1,1]))
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            idx = np.random.randint(0, len(self.iaanoise))
            tran = self.iaanoise[idx]
            im = sample['image'].astype('uint8')
            im = tran(image=im)
            sample['image'] = im.astype('float32')
        return sample

class IAABlur(BaseOperator):
    def __init__(self, prob=0.5):
        super(IAABlur, self).__init__()

        self.iaablur = []
        self.iaablur.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))
        self.iaablur.append(iaa.imgcorruptlike.DefocusBlur(severity=[1,1,1]))
        self.iaablur.append(iaa.imgcorruptlike.MotionBlur(severity=[1,1,1]))
        # 这个效果太猛烈了，所以注释
        # self.iaablur.append(iaa.imgcorruptlike.ZoomBlur(severity=[1,1,1]))
        self.iaablur.append(iaa.imgcorruptlike.Spatter(severity=[1,1,1]))
        self.iaablur.append(iaa.imgcorruptlike.Pixelate(severity=[1,1,1]))
        self.iaablur.append(iaa.imgcorruptlike.ElasticTransform(severity=[1,1,1]))
        self.iaablur.append(iaa.MedianBlur(k=(3, 11)))
        self.iaablur.append(iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0)))
   
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            idx = np.random.randint(0, len(self.iaablur))
            tran = self.iaablur[idx]
            im = sample['image'].astype('uint8')
            im = tran(image=im)
            sample['image'] = im.astype('float32')
        return sample

class IAAEmboss(BaseOperator):
    def __init__(self, prob=0.5,):
        super(IAAEmboss, self).__init__()
        self.Emboss = iaa.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7))
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            cv_img = sample['image'].astype('uint8')
            cv_img = self.Emboss(image=cv_img)
            sample['image'] = cv_img.astype('float32')
        return sample
    
class CLAHE(BaseOperator):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), prob=0.5,):
        super(CLAHE, self).__init__()
        self.clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            cv_img = sample['image'].astype('uint8')
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
            cv_img[:, :, 0] = self.clahe_mat.apply(cv_img[:, :, 0])
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_LAB2RGB)
            sample['image'] = cv_img.astype('float32')
        return sample
    
class Hist(BaseOperator):
    def __init__(self, prob=0.5):
        super(Hist, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            image = sample['image'].astype('uint8')
            r, g, b = cv2.split(image)
            r1 = cv2.equalizeHist(r)
            g1 = cv2.equalizeHist(g)
            b1 = cv2.equalizeHist(b)
            image_equal_clo = cv2.merge([r1, g1, b1])
            sample['image'] = image_equal_clo.astype('float32')
        return sample
    
class IAAElastic(BaseOperator):
    def __init__(self, prob=0.5):
        super(IAAElastic, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            tran = iaa.OneOf([
                    iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0)),
                    iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                    iaa.Cutout(fill_mode="constant", cval=(0, 255),fill_per_channel=0.5),
                    # iaa.RandAugment(n=2, m=9),
                    # iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256)),
                    # iaa.Jigsaw(nb_rows=10, nb_cols=10),
                    iaa.BlendAlpha((0.0, 1.0),foreground=iaa.Add(100),background=iaa.Multiply(0.2))
                    ])
            im = sample['image'].astype('uint8')
            im = tran(image=im)
            sample['image'] = im.astype('float32')
        return sample

class IAAColor(BaseOperator):
    def __init__(self, prob=0.5):
        super(IAAColor, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            tran = iaa.OneOf([
                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                    iaa.MultiplyHue((0.5, 1.5)),
                    iaa.UniformColorQuantization(),
                    # 不要用这个，破坏性太大
                    # iaa.KMeansColorQuantization(from_colorspace=iaa.ChangeColorspace.BGR),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
                    iaa.imgcorruptlike.Saturate(severity=[2,3,4]),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)
                    ])
            im = sample['image'].astype('uint8')
            im = tran(image=im)
            sample['image'] = im.astype('float32')
        return sample

class IAAWeather(BaseOperator):
    def __init__(self, prob=0.5):
        super(IAAWeather, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            tran = iaa.OneOf([
                # 云不要了，不太符合实际情况
                    # iaa.Clouds(),
                    iaa.FastSnowyLandscape(lightness_threshold=[128, 200],lightness_multiplier=(1.5, 3.5)),
                    iaa.Fog(),
                    iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
                    iaa.Rain(speed=(0.1, 0.3)),
                    iaa.imgcorruptlike.Frost(severity=[1,1,1])
                    ])
            im = sample['image'].astype('uint8')
            im = tran(image=im)
            sample['image'] = im.astype('float32')
        return sample
    
class Lighting(BaseOperator):
    """
    Lighting the image by eigenvalues and eigenvectors
    Args:
        eigval (list): eigenvalues
        eigvec (list): eigenvectors
        alphastd (float): random weight of lighting, 0.1 by default
    """

    def __init__(self, eigval, eigvec, alphastd=0.1):
        super(Lighting, self).__init__()
        self.alphastd = alphastd
        self.eigval = np.array(eigval).astype('float32')
        self.eigvec = np.array(eigvec).astype('float32')

    def apply(self, sample, context=None):
        alpha = np.random.normal(scale=self.alphastd, size=(3, ))
        sample['image'] += np.dot(self.eigvec, self.eigval * alpha)
        return sample

class RandomErasingImage(BaseOperator):
    def __init__(self, prob=0.5, lower=0.02, higher=0.4, aspect_ratio=0.3):
        """
        Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
        Args:
            prob (float): probability to carry out random erasing
            lower (float): lower limit of the erasing area ratio
            higher (float): upper limit of the erasing area ratio
            aspect_ratio (float): aspect ratio of the erasing region
        """
        super(RandomErasingImage, self).__init__()
        self.prob = prob
        self.lower = lower
        self.higher = higher
        self.aspect_ratio = aspect_ratio

    def apply(self, sample, context=None):
        gt_bbox = sample['gt_bbox']
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image is not a numpy array.".format(self))
        if len(im.shape) != 3:
            raise ImageError("{}: image is not 3-dimensional.".format(self))

        for idx in range(gt_bbox.shape[0]):
            if self.prob <= np.random.rand():
                continue

            x1, y1, x2, y2 = gt_bbox[idx, :]
            w_bbox = x2 - x1
            h_bbox = y2 - y1
            area = w_bbox * h_bbox

            target_area = random.uniform(self.lower, self.higher) * area
            aspect_ratio = random.uniform(self.aspect_ratio,
                                          1 / self.aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < w_bbox and h < h_bbox:
                off_y1 = random.randint(0, int(h_bbox - h))
                off_x1 = random.randint(0, int(w_bbox - w))
                im[int(y1 + off_y1):int(y1 + off_y1 + h), int(x1 + off_x1):int(
                    x1 + off_x1 + w), :] = 0
        sample['image'] = im
        return sample

class GridMask(BaseOperator):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.8,
                 mode=1,
                 prob=0.5,
                 upper_iter=36000):
        """
        GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086
        Args:
            use_h (bool): whether to mask vertically
            use_w (boo;): whether to mask horizontally
            rotate (float): angle for the mask to rotate
            offset (float): mask offset
            ratio (float): mask ratio
            mode (int): gridmask mode
            prob (float): max probability to carry out gridmask
            upper_iter (int): suggested to be equal to global max_iter
        """
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.upper_iter = upper_iter

        from .gridmask_utils import Gridmask
        self.gridmask_op = Gridmask(
            use_h,
            use_w,
            rotate=rotate,
            offset=offset,
            ratio=ratio,
            mode=mode,
            prob=prob,
            upper_iter=upper_iter)

    def apply(self, sample, context=None):
        sample['image'] = self.gridmask_op(sample['image'], sample['curr_iter'])
        # if random.random() < 0.5:
        #     filename = str(time.time())+'.png'
        #     filepath = "/mnt/output/vis_out/" + filename
        #     cv2.imwrite(filename=filepath, img=sample['image'][:,:,::-1])
        return sample
    
class BBoxJitter(BaseOperator):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min=0.95, max=1.05):
        super(BBoxJitter, self).__init__()
        self.min_scale = min
        self.max_scale = max
        self.count = 0

    def bbox_jitter(self, bboxes, img):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        height, width = img.shape[:2]
        img_shape = []
        img_shape.append(height)
        img_shape.append(width)
        # print(img_shape)
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def apply(self, sample, context=None):
        im = sample['image']
        sample['gt_bbox'] = self.bbox_jitter(sample['gt_bbox'], im)
        return sample
    

class HSVAugment(BaseOperator):
    """
    Color Gamut to img

    Args:
        hgain: The gain of hues channel.
        sgain: The gain of saturation channel.
        vgain: The gain of brightness channel.
    """
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        super(HSVAugment, self).__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply(self, sample, context=None):
        cv_img = sample['image'].astype('uint8')

        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV))
        dtype = cv_img.dtype  # type

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=cv_img.astype(np.float32))  # no return needed
        sample['image'] = cv_img.astype('float32')
        return sample
    
    
class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1], is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.is_scale = is_scale
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']

        # 用来绘制可视化图片的，正式训练的时候需要注释
        # gt_bbox = sample['gt_bbox']
        # for i in range(gt_bbox.shape[0]):
        #     xmin, ymin, xmax, ymax = gt_bbox[i]
        #     cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # save_dir = 'vis_1'
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # filename = str(time.time()) + '.png'
        # filepath = os.path.join(save_dir, filename)
        # cv2.imwrite(filepath, im[:,:,::-1])
        # ---------------------------------------------------------

        
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std

        sample['image'] = im
        return sample


class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                    width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(
                    sample['gt_keypoint'], width)

            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]

            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                         width)

            sample['flipped'] = True
            sample['image'] = im
        return sample


class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample



class RandomDistort(BaseOperator):
    """
    Distort an image with random configurations.

    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def apply(self, data,  context=None):

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.sharpness
        ]
        if data['image'].ndim > 2:
            ops.append(functional.hue)
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        data['image'] = data['image'].astype('uint8')
        data['image'] = Image.fromarray(data['image'])
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = data['image']
            if np.random.uniform(0, 1) < prob:
                data['image'] = ops[id](**params)
        data['image'] = np.asarray(data['image']).astype('float32')
        return data

class AutoAugment(BaseOperator):
    def __init__(self, autoaug_type="v1"):
        """
        Args:
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(AutoAugment, self).__init__()
        self.autoaug_type = autoaug_type

    def apply(self, sample, context=None):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image is not a numpy array.".format(self))
        if len(im.shape) != 3:
            raise ImageError("{}: image is not 3-dimensional.".format(self))
        if len(gt_bbox) == 0:
            return sample

        height, width, _ = im.shape
        norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
        norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
        norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
        norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
        norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)

        from .autoaugment_utils import distort_image_with_autoaugment
        im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
                                                          self.autoaug_type)

        gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
        gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
        gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
        gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)

        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        return sample

class MultiscaleTestResize(BaseOperator):
    def __init__(self,
                 origin_target_size=[800, 1333],
                 target_size=[],
                 interp=cv2.INTER_LINEAR,
                 use_flip=True):
        """
        Rescale image to the each size in target size, and capped at max_size.
        Args:
            origin_target_size (list): origin target size of image
            target_size (list): A list of target sizes of image.
            interp (int): the interpolation method.
            use_flip (bool): whether use flip augmentation.
        """
        super(MultiscaleTestResize, self).__init__()
        self.interp = interp
        self.use_flip = use_flip

        if not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid. Must be List or Tuple, now is {}".
                format(type(target_size)))
        self.target_size = target_size

        if not isinstance(origin_target_size, Sequence):
            raise TypeError(
                "Type of origin_target_size is invalid. Must be List or Tuple, now is {}".
                format(type(origin_target_size)))

        self.origin_target_size = origin_target_size

    def apply(self, sample, context=None):
        """ Resize the image numpy for multi-scale test.
        """
        samples = []
        resizer = Resize(
            self.origin_target_size, keep_ratio=True, interp=self.interp)
        samples.append(resizer(sample.copy(), context))
        if self.use_flip:
            flipper = RandomFlip(1.1)
            samples.append(flipper(sample.copy(), context=context))

        for size in self.target_size:
            resizer = Resize(size, keep_ratio=True, interp=self.interp)
            samples.append(resizer(sample.copy(), context))

        return samples

class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=1.1, prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value, ) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def apply(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size,
                  pad_mode=-1,
                  offsets=offsets,
                  fill_value=self.fill_value)

        return pad(sample, context=context)
    
class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def apply(self, sample, context):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        height, width, _ = im.shape
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample

class CropWithDataAchorSampling(BaseOperator):
    def __init__(self,
                 batch_sampler,
                 anchor_sampler=None,
                 target_size=None,
                 das_anchor_scales=[16, 32, 64, 128],
                 sampling_prob=0.5,
                 min_size=8.,
                 avoid_no_bbox=True):
        """
        Args:
            anchor_sampler (list): anchor_sampling sets of different
                                  parameters for cropping.
            batch_sampler (list): Multiple sets of different
                                  parameters for cropping.
              e.g.[[1, 10, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.0]]
                  [[1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]
              [max sample, max trial, min scale, max scale,
               min aspect ratio, max aspect ratio,
               min overlap, max overlap, min coverage, max coverage]
            target_size (int): target image size.
            das_anchor_scales (list[float]): a list of anchor scales in data
                anchor smapling.
            min_size (float): minimum size of sampled bbox.
            avoid_no_bbox (bool): whether to avoid the
                                  situation where the box does not appear.
        """
        super(CropWithDataAchorSampling, self).__init__()
        self.anchor_sampler = anchor_sampler
        self.batch_sampler = batch_sampler
        self.target_size = target_size
        self.sampling_prob = sampling_prob
        self.min_size = min_size
        self.avoid_no_bbox = avoid_no_bbox
        self.das_anchor_scales = np.array(das_anchor_scales)

    def apply(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        image_height, image_width = im.shape[:2]
        gt_bbox[:, 0] /= image_width
        gt_bbox[:, 1] /= image_height
        gt_bbox[:, 2] /= image_width
        gt_bbox[:, 3] /= image_height
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()

        prob = np.random.uniform(0., 1.)
        if prob > self.sampling_prob:  # anchor sampling
            assert self.anchor_sampler
            for sampler in self.anchor_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = data_anchor_sampling(
                        gt_bbox, image_width, image_height,
                        self.das_anchor_scales, self.target_size)
                    if sample_bbox == 0:
                        break
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                scores=gt_score,
                                keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                im = crop_image_sampling(im, sample_bbox, image_width,
                                         image_height, self.target_size)
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
                    sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample

        else:
            for sampler in self.batch_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = generate_sample_bbox_square(
                        sampler, image_width, image_height)
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)
                sample_bbox = clip_bbox(sample_bbox)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                scores=gt_score,
                                keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                # sampling bbox according the bbox area
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                xmin = int(sample_bbox[0] * image_width)
                xmax = int(sample_bbox[2] * image_width)
                ymin = int(sample_bbox[1] * image_height)
                ymax = int(sample_bbox[3] * image_height)
                im = im[ymin:ymax, xmin:xmax]
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
                    sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample

class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def crop_segms(self, segms, valid_ids, crop, height, width):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                logging.getLogger("shapely").setLevel(logging.WARNING)
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h, w = sample['image'].shape[:2]
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale**2), min(max_ar, scale**-2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_poly' in sample and len(sample[
                        'gt_poly']) > 0:
                    crop_polys = self.crop_segms(
                        sample['gt_poly'],
                        valid_ids,
                        np.array(
                            crop_box, dtype=np.int64),
                        h,
                        w)
                    if [] in crop_polys:
                        delete_id = list()
                        valid_polys = list()
                        for id, crop_poly in enumerate(crop_polys):
                            if crop_poly == []:
                                delete_id.append(id)
                            else:
                                valid_polys.append(crop_poly)
                        valid_ids = np.delete(valid_ids, delete_id)
                        if len(valid_polys) == 0:
                            return sample
                        sample['gt_poly'] = valid_polys
                    else:
                        sample['gt_poly'] = crop_polys

                if 'gt_segm' in sample:
                    sample['gt_segm'] = self._crop_segm(sample['gt_segm'],
                                                        crop_box)
                    sample['gt_segm'] = np.take(
                        sample['gt_segm'], valid_ids, axis=0)

                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

                if 'difficult' in sample:
                    sample['difficult'] = np.take(
                        sample['difficult'], valid_ids, axis=0)

                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

    def _crop_segm(self, segm, crop):
        x1, y1, x2, y2 = crop
        return segm[:, y1:y2, x1:x2]


class RandomScaledCrop(BaseOperator):
    """Resize image and bbox based on long side (with optional random scaling),
       then crop or pad image to target size.
    Args:
        target_dim (int): target size.
        scale_range (list): random scale range.
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self,
                 target_dim=512,
                 scale_range=[.1, 2.],
                 interp=cv2.INTER_LINEAR):
        super(RandomScaledCrop, self).__init__()
        self.target_dim = target_dim
        self.scale_range = scale_range
        self.interp = interp

    def apply(self, sample, context=None):
        img = sample['image']
        h, w = img.shape[:2]
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim
        random_dim = int(dim * random_scale)
        dim_max = max(h, w)
        scale = random_dim / dim_max
        resize_w = w * scale
        resize_h = h * scale
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))

        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
            offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['image'] = canvas
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        scale_factor = sample['sacle_factor']
        sample['scale_factor'] = np.asarray(
            [scale_factor[0] * scale, scale_factor[1] * scale],
            dtype=np.float32)

        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            shift_array = np.array([offset_x, offset_y] * 2, dtype=np.float32)
            boxes = sample['gt_bbox'] * scale_array - shift_array
            boxes = np.clip(boxes, 0, dim - 1)
            # filter boxes with no area
            area = np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)
            valid = (area > 1.).nonzero()[0]
            sample['gt_bbox'] = boxes[valid]
            sample['gt_class'] = sample['gt_class'][valid]

        return sample


class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class RandomSelect(BaseOperator):
    """
    Randomly choose a transformation between transforms1 and transforms2,
    and the probability of choosing transforms1 is p.

    The code is based on https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py

    """

    def __init__(self, transforms1, transforms2, p=0.5):
        super(RandomSelect, self).__init__()
        self.transforms1 = Compose(transforms1)
        self.transforms2 = Compose(transforms2)
        self.p = p

    def apply(self, sample, context=None):
        if random.random() < self.p:
            return self.transforms1(sample)
        return self.transforms2(sample)


class RandomShortSideResize(BaseOperator):
    def __init__(self,
                 short_side_sizes,
                 max_size=None,
                 interp=cv2.INTER_LINEAR,
                 random_interp=False):
        """
        Resize the image randomly according to the short side. If max_size is not None,
        the long side is scaled according to max_size. The whole process will be keep ratio.
        Args:
            short_side_sizes (list|tuple): Image target short side size.
            max_size (int): The size of the longest side of image after resize.
            interp (int): The interpolation method.
            random_interp (bool): Whether random select interpolation method.
        """
        super(RandomShortSideResize, self).__init__()

        assert isinstance(short_side_sizes,
                          Sequence), "short_side_sizes must be List or Tuple"

        self.short_side_sizes = short_side_sizes
        self.max_size = max_size
        self.interp = interp
        self.random_interp = random_interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

    def get_size_with_aspect_ratio(self, image_shape, size, max_size=None):
        h, w = image_shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def resize(self,
               sample,
               target_size,
               max_size=None,
               interp=cv2.INTER_LINEAR):
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        target_size = self.get_size_with_aspect_ratio(im.shape[:2], target_size,
                                                      max_size)
        im_scale_y, im_scale_x = target_size[1] / im.shape[0], target_size[
            0] / im.shape[1]

        sample['image'] = cv2.resize(im, target_size, interpolation=interp)
        sample['im_shape'] = np.asarray(target_size[::-1], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im.shape[:2],
                                                [im_scale_x, im_scale_y])
        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                target_size,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm, target_size, interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox.astype('float32')

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        target_size = random.choice(self.short_side_sizes)
        interp = random.choice(
            self.interps) if self.random_interp else self.interp

        return self.resize(sample, target_size, self.max_size, interp)


class RandomSizeCrop(BaseOperator):
    """
    Cut the image randomly according to `min_size` and `max_size`
    """

    def __init__(self, min_size, max_size):
        super(RandomSizeCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

        from paddle.vision.transforms.functional import crop as paddle_crop
        self.paddle_crop = paddle_crop

    @staticmethod
    def get_crop_params(img_shape, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img_shape (list|tuple): Image's height and width.
            output_size (list|tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (h, w)))

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        return i, j, th, tw

    def crop(self, sample, region):
        image_shape = sample['image'].shape[:2]
        sample['image'] = self.paddle_crop(sample['image'], *region)

        keep_index = None
        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], region)
            bbox = sample['gt_bbox'].reshape([-1, 2, 2])
            area = (bbox[:, 1, :] - bbox[:, 0, :]).prod(axis=1)
            keep_index = np.where(area > 0)[0]
            sample['gt_bbox'] = sample['gt_bbox'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 4], dtype=np.float32)
            sample['gt_class'] = sample['gt_class'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 1], dtype=np.float32)
            if 'gt_score' in sample:
                sample['gt_score'] = sample['gt_score'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)
            if 'is_crowd' in sample:
                sample['is_crowd'] = sample['is_crowd'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], region,
                                                image_shape)
            if keep_index is not None:
                sample['gt_poly'] = sample['gt_poly'][keep_index]
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            i, j, h, w = region
            sample['gt_segm'] = sample['gt_segm'][:, i:i + h, j:j + w]
            if keep_index is not None:
                sample['gt_segm'] = sample['gt_segm'][keep_index]

        return sample

    def apply_bbox(self, bbox, region):
        i, j, h, w = region
        region_size = np.asarray([w, h])
        crop_bbox = bbox - np.asarray([j, i, j, i])
        crop_bbox = np.minimum(crop_bbox.reshape([-1, 2, 2]), region_size)
        crop_bbox = crop_bbox.clip(min=0)
        return crop_bbox.reshape([-1, 4]).astype('float32')

    def apply_segm(self, segms, region, image_shape):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        i, j, h, w = region
        crop = [j, i, j + w, i + h]
        height, width = image_shape
        crop_segms = []
        for segm in segms:
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        h = random.randint(self.min_size,
                           min(sample['image'].shape[0], self.max_size))
        w = random.randint(self.min_size,
                           min(sample['image'].shape[1], self.max_size))

        region = self.get_crop_params(sample['image'].shape[:2], [h, w])
        return self.crop(sample, region)


class PadMaskBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
        return_pad_mask (bool): If `return_pad_mask = True`, return
            `pad_mask` for transformer.
    """

    def __init__(self, pad_to_stride=0, return_pad_mask=False):
        super(PadMaskBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.return_pad_mask = return_pad_mask

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm
            if self.return_pad_mask:
                padding_mask = np.zeros(
                    (max_shape[1], max_shape[2]), dtype=np.float32)
                padding_mask[:im_h, :im_w] = 1.
                data['pad_mask'] = padding_mask

            if 'gt_rbox2poly' in data and data['gt_rbox2poly'] is not None:
                # ploy to rbox
                polys = data['gt_rbox2poly']
                rbox = bbox_utils.poly2rbox(polys)
                data['gt_rbox'] = rbox

        return samples
    

class Cutmix(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ 
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(Cutmix, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int32(w * cut_rat)
        cut_h = np.int32(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w - 1)
        bby1 = np.clip(cy - cut_h // 2, 0, h - 1)
        bbx2 = np.clip(cx + cut_w // 2, 0, w - 1)
        bby2 = np.clip(cy + cut_h // 2, 0, h - 1)

        img_1_pad = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1_pad[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32')
        img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2_pad[:img2.shape[0], :img2.shape[1], :] = \
            img2.astype('float32')
        img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
        return img_1_pad

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'cutmix need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        img1 = sample[0]['image']
        img2 = sample[1]['image']
        img = self.apply_image(img1, img2, factor)
        gt_bbox1 = sample[0]['gt_bbox']
        gt_bbox2 = sample[1]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample[0]['gt_class']
        gt_class2 = sample[1]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = np.ones_like(sample[0]['gt_class'])
        gt_score2 = np.ones_like(sample[1]['gt_class'])
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        result = copy.deepcopy(sample[0])
        result['image'] = img
        result['gt_bbox'] = gt_bbox
        result['gt_score'] = gt_score
        result['gt_class'] = gt_class
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult
        return result

class Pad_changeimshape(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad_changeimshape, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_segm(self, segms, offsets, im_size, size):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, h, w):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((h, w), 0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_keypoint(self, keypoints, offsets):
        n = len(keypoints[0]) // 2
        return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply(self, sample, context=None):
        im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            sample['image'] = im.astype(np.float32)
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['image'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            ## insert code 
            assert self.size[0] == self.size[1]
            origin_shape = sample['im_shape'] / sample['scale_factor']
            max_length = max(origin_shape)

            #update sample['im_shape']
            sample['im_shape'] = np.array(sample['image'].shape[:2], dtype=np.float32)

            #update scale
            scale = np.array(
                (sample['im_shape'][0] / max_length, sample['im_shape'][1] / max_length), 
                dtype=np.float32)
            sample['scale_factor'] = scale

            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets,
                                                im_size, size)

        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'],
                                                        offsets)
        return sample