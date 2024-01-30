from fastai.vision.all import *
import albumentations
import torchvision
import torchvision.transforms as transforms

class AlbumentationsTransform(DisplayedTransform):
    '''
    Class that allows the use of Albumentations transforms in FastAI.
    '''

    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def create_learner(df, model, hyperparameters, metrics, seed=18):
    '''
    Creates a FastAI learner for a single label classification task.

    Args:
        df (Pandas.DataFrame): dataframe containing the information of each entry of the dataset (with the columns 'file_path', 'label' and 'is_valid').
        model (function or str): architecture of the model, could be either a 'torchvision' model or a 'timm' model.
        hyperparameters (dict): dictionary containing the hyperparameters of the model, contains 'IMG_SIZE', 'BS', 'TRANSFORMS' and 'WD'.
        metrics (list): list of metrics that will be used to evaluate the model.
        seed (int): seed setted for reproducibility (default 18).

    Returns:
        learn (fastai.learner.Learner): FastAI learner for training a model for the single label classification task.
    '''

    set_seed(seed, True)
    # Create datablock
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('file_path'),
        get_y=ColReader('label'),
        splitter=ColSplitter(col='is_valid'),
        item_tfms=[Resize(hyperparameters['IMG_SIZE'], method='squish'), AlbumentationsTransform(albumentations.Compose(hyperparameters['TRANSFORMS']))])
    # Create dataloaders
    dls = block.dataloaders(df, bs=hyperparameters['BS'], shuffle=True)
    dls.rng.seed(seed)
    # Explore classes
    num_classes = dls.c
    classes = dls.vocab
    print('Number of clases: ', num_classes)
    print('Names of classes: ', classes)
    # Create model
    learn = vision_learner(dls,
                        model,
                        normalize=True,
                        pretrained=True,
                        opt_func=Adam,
                        metrics=metrics,
                        wd=hyperparameters['WD']).to_fp16()
    return learn


def get_predictions_table(learn, dl):
    '''
    Creates a table containing a row for each image stored in 'dl'.

    Args:
        learn (fastai.learner.Learner): trained learner object.
        dl (fastai.data.core.TfmdDL): dataloader with the images for making predictions.

    Returns:
        df (pandas.core.frame.DataFrame): table with columns=['file_name', 'ground_truth', 'prediction', 'loss', 'confidence'], sorted by 'loss' value in descending order.
    '''

    labels = dl.vocab
    file_paths = dl.dataset.items.file_path.values
    probs, ground_truths, losses = learn.get_preds(dl=dl, with_loss=True)
    predictions = np.argmax(probs, axis=1)
    data = np.array([file_paths, np.array(labels[ground_truths]), np.array(labels[predictions]), np.array(losses), np.max(probs.numpy(),axis=1)]).T
    table = pd.DataFrame(data=data, columns=["file_name", "ground_truth", "prediction", "loss", "confidence"])
    
    return table.sort_values(by='loss', ascending=False)


def fastai2onnx(learn, path, height, width):
    '''
    Converts a FastAI model from a Learner object to an ONNX model, and then saves it to disk.

    Args:
        learn (fastai.learner.Learner): trained learner object.
        path (str): folder where to save the 'model.onnx' file.
        height: input height for the model.
        width: input width for the model.
    '''

    # Extract pytorch model from the learner and add missing layers
    softmax_layer = torch.nn.Softmax(dim=1)
    normalization_layer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_model = nn.Sequential(
        normalization_layer,
        learn.model.eval(),
        softmax_layer
    )

    # Export from pytorch to onnx 
    torch.onnx.export(
        pytorch_model,
        torch.randn(1, 3, height, width).to('cuda'),
        f'{path}/model.onnx',
        do_constant_folding=True,
        export_params=True,
        input_names=['in'],
        output_names=['out'],
        opset_version=17,
        dynamic_axes={'in' : {0 : 'batch_size'},    # variable length axes
                    'out' : {0 : 'batch_size'}}
    )    
