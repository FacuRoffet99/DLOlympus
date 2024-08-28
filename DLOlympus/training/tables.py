from fastai.vision.all import *
import albumentations
import torchvision
import torchvision.transforms as transforms


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



