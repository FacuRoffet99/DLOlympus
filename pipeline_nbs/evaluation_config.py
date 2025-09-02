import wandb
from DLOlympus.utils.plots import plot_confusion_matrix

def create_predictions_df(learn, dl, split):
	raw_outputs, _, outputs, losses = learn.get_preds(dl=dl, with_decoded=True, with_loss=True)
    
	df = dl.items.drop(columns=['is_valid'])
	df['file_path'] = df['file_path'].apply(lambda x: x.split(split)[-1])
	df['pred_breed'] = dl.vocab[0][outputs[0]]
	df['score_breed'] = raw_outputs[0].max(dim=-1).values
	df['pred_animal'] = dl.vocab[1][outputs[1]]
	df['score_animal'] = raw_outputs[1].max(dim=-1).values
	df['pred_group'] = outputs[2]
	df['loss'] = losses

	return df.sort_values(by='loss', ascending=False)

def create_confusion_matrices(valid_preds, vocabs):
	gts = ['label_breed', 'label_animal']
	preds = ['pred_breed', 'pred_animal']

	names_cms = ['cm_breeds', 'cm_animals']

	plt_cms = [plot_confusion_matrix(valid_preds[g].values, valid_preds[p].values, v) for g,p,v in zip(gts,preds,vocabs)]
	wandb_cms = [wandb.plot.confusion_matrix(preds=[v.o2i[i] for i in valid_preds[p].values],
									  	y_true=[v.o2i[i] for i in valid_preds[g].values],
                            			class_names=v,
										title='Confusion matrix') 
										for g,p,v in zip(gts,preds,vocabs)]

	return plt_cms, wandb_cms, names_cms