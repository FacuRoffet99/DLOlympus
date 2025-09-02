import numpy as np


def get_metrics(log_file):
	# Read log
	with open(log_file) as f:
		lines = f.readlines()
	# Select validation lines
	filtered_lines = [
		[metric for metric in line.split('	')[1].split('  ') if metric.startswith('coco/')]
		for line in lines if 'coco/bbox_mAP:' in line
	]
	# Get metric names
	metrics_names = [metric.split('/')[1].split(':')[0] for metric in filtered_lines[0]]
	# Get metrics values (n_epochs x n_metrics)
	metrics_values = np.array([
		[float(metric.split(': ')[1]) for metric in metrics]
		for metrics in filtered_lines
	])
	# Return only final values
	best_epoch = int([lin for lin in lines if 'best' in lin][-1].split('_')[-1].split('.')[0])
	metrics = {k: v.item() for k,v in zip(metrics_names, metrics_values[best_epoch-1])}
	return metrics