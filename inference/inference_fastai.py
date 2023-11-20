from fastai.vision.all import *
from PIL import ImageDraw, ImageFont
import pathlib
from .inference_base import ModelInferencer
from .fix import load_learner

class FastAIInferencer(ModelInferencer):
    '''
    Base (abstract) class for implementing an inferencer with FastAI.

    Args:
        checkpoint_file (str): Path to the .pkl file where the model weights are stored.
        labels (list): List containing the names of the classes to predict (in order). 
    '''

    def init(self, cpu=True):
        '''
        Initialization method to create a learner with the model weights an configs.

        Args:
            cpu (bool): If 'True' the inferencer will run in cpu, if False it will run in gpu (if available).
        '''
        self.learn = load_learner(self.checkpoint_file, cpu=cpu)
    
    def set_params(self, param_dict):
        print('This model does not use any parameters.')    



class FastAIClassifierInferencer(FastAIInferencer):
    '''
    Class for implementing an inferencer for an image classifier trained with FastAI.

    Args:
        checkpoint_file (str): Path to the .pkl file where the model weights are stored.
        labels (list): List containing the names of the classes to predict (in order). 
    '''
        
    def process(self, imgs):
        '''
        Process images for getting the predictions results.

        Args:
            imgs (list): The source of the images to process. Could be either a list of paths or a list of images as numpy arrays.

        Returns:
            results (dict): The prediction results. The values of the dict are tuples containing the name and the probability of the predicted class.
        '''
        test_dl = self.learn.dls.test_dl(imgs)
        probs, _ = self.learn.get_preds(dl=test_dl)
        results = {}
        for i, prob in enumerate(probs):
            results[i] = (self.labels[prob.argmax()], prob.max().item())
        return results



class FastAIKeypointsRegressorInferencer(FastAIInferencer):
    '''
    Class for implementing an inferencer for an image keypoint regressor trained with FastAI.

    Args:
        checkpoint_file (str): Path to the .pkl file where the model weights are stored.
        labels (list): List containing the names of the classes to predict (in order). 
    '''

    def set_params(self, param_dict):
        self.width = param_dict['width']
        self.height = param_dict['height']

    def process(self, imgs):
        '''
        Process images for getting the predictions results.

        Args:
            imgs (list): The source of the images to process. Could be either a list of paths or a list of images as numpy arrays.

        Returns:
            results (dict): The prediction results. The values of the dict are numpy arrays containing the coordinates of each predicted point.
        '''
        results = {}
        for i, img in enumerate(imgs):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img).resize((self.width, self.height))
            elif isinstance(img, str) or isinstance(img, pathlib.PosixPath):
                img = Image.open(img).resize((self.width, self.height))
            results[i] = self.learn.predict(img)[0]
        self.imgs = imgs
        self.results = results
        return results

    def draw_prediction(self, i, size=4, font_size=15):
        '''
        Draws the predicted keypoints of an image over it.

        Args:
            i (int): The index of the image to use.
            size (int): Radius of the points that will be drawn (default: 4).
            font_size (int): Size of the font for the annotations (default: 15).

        Returns:
            temp (PIL.Image): The annotated image.
        '''
        if isinstance(self.imgs[i], np.ndarray):
            img = Image.fromarray(self.imgs[i]).resize((self.width, self.height))
        elif isinstance(self.imgs[i], str) or isinstance(self.imgs[i], pathlib.PosixPath):
            img = Image.open(self.imgs[i]).resize((self.width, self.height))
        keypoints = self.results[i]
        temp = img.copy()
        draw = ImageDraw.Draw(temp)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        if keypoints.ndim == 1:
            kp = keypoints
            draw.ellipse((kp[0]-size, kp[1]-size, kp[0]+size, kp[1]+size), fill=(255,0,0))
        else:
            for tn, kp in zip(self.labels, keypoints):
                draw.ellipse((kp[0]-size, kp[1]-size, kp[0]+size, kp[1]+size), fill=(255,0,0))
                draw.text((kp[0], kp[1]), tn, font=font, fill='black', anchor='ma')
        return temp
