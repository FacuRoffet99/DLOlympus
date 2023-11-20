import os
import numpy as np
import mmcv
import torch
from PIL import Image
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.core.visualization import imshow_det_bboxes
from .inference_base import ModelInferencer

def res2pil(img):
    b = np.copy(img[:,:,0])
    r = np.copy(img[:,:,2])
    img[:,:,2] = b
    img[:,:,0] = r
    return Image.fromarray(img)      

class MMDetectionSegmentatorInferencer(ModelInferencer):
    '''
    Class for implementing an inferencer for an image segmentation model trained with MMDetection.

    Args:
        checkpoint_file (str): Path to the .pth file where the model weights are stored.
        labels (list): List containing the names of the classes to predict (in order). 
    '''

    def _edit_config(self, img_folder):
        '''
        Update the configurations file with the images that will be processed.

        Args:
        img_folder (str): Path to the folder where the images are stored.
        '''
        cfg = Config.fromfile(self.config)
        cfg.data.test.img_prefix = img_folder
        cfg.dump(self.config)

    def init(self, config_file, cpu=True):
        '''
        Initialization method.

        Args:
            config_file (str): Path to the configuration file.
            cpu (bool): If 'True' the inferencer will run in cpu, if False it will run in gpu (if available).
        '''
        self.config = config_file
        self.cpu = cpu
    
    def set_params(self, param_dict):
        self.detection_threshold = param_dict['detection_threshold']
        self.img_max = param_dict['img_max']

    def process(self, img_folder):
        '''
        Process images for getting the predictions results.

        Args:
            img_folder (str): The source of the images to process. For now, it can only be a folder with images.

        Returns:
            results (dict): The prediction results. The values of the dict are tuples containing the following information of each detected object:
                *The class name.
                *The coordinates of the bounding box and the probability of the prediction.
                *A binary segmentation mask.
        '''
        device = 'cpu' if self.cpu else 'cuda'
        img_extensions = ('.jpg', '.jpeg', '.png')
        imgs = ['/'.join((img_folder, img_name)) for img_name in os.listdir(img_folder) if img_name.endswith(img_extensions)]
        
        to_process = imgs
        preds_raw = []
        while len(to_process) > 0:
            # Select data
            img_paths = to_process if len(to_process)<self.img_max else to_process[:self.img_max]
            # Prepare the data
            self._edit_config(img_folder)
            # Build the model
            self.model = init_detector(self.config, self.checkpoint_file, device=device)
            # Get the predictions
            preds_raw += inference_detector(self.model, img_paths)
            # Finish if all imgs had been processed
            if len(to_process) < self.img_max: break
            # Remove used data
            to_process = to_process[self.img_max:]

        # Format the results in a more readable form
        results = {}
        for i, pred in enumerate(preds_raw):     
            bbox_result, segm_result = pred
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
            bboxes = np.vstack(bbox_result)
            classes = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
            classes = np.concatenate(classes)
            classes = np.array(self.labels)[classes]
            segms = None
            if segm_result is not None and len(classes) > 0:
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                else:
                    segms = np.stack(segms, axis=0)

            # Filter results according to the threshold
            to_keep = [i for i, bbox in enumerate(bboxes) if bbox[4] >= self.detection_threshold]
            classes, bboxes, segms = classes[to_keep], bboxes[to_keep], segms[to_keep]

            results[i] = (classes, bboxes, segms)

        self.imgs = imgs
        self.results = results
        return results

    def draw_prediction(self, i, thickness=2, font_size=13):
        '''
        Draws the predicted segmentations of an image over it.

        Args:
            i (int): The index of the image to use.
            thickness (int): Thickness of the bounding boxes that will be drawn (default: 2).
            font_size (int): Size of the font for the annotations (default: 13).

        Returns:
            temp (PIL.Image): The annotated image.
        '''
        img = mmcv.imread(self.imgs[i])
        img = img.copy()
        pred_names, bboxes, segms = self.results[i]
        pred_ids = np.array([self.labels.index(i) for i in pred_names])
        img = imshow_det_bboxes(
            img,
            bboxes,
            pred_ids,
            segms,
            class_names=self.labels,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=None,
            thickness=thickness,
            font_size=font_size,
            show=False)   
        return res2pil(img)   

