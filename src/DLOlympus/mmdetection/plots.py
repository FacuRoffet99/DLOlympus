import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(log_file, path):
	# Read log
	with open(log_file) as f:
		lines = f.readlines()
	# Select and trim training lines
	train_losses = np.array([
		float(line.split('loss:')[1].split()[0])
		for line in lines if 'Epoch(train)' in line
	])
	# Select and trim validation lines
	valid_losses = np.array([
		float(line.split(': ')[-1].split('\n')[0])
		for line in lines if 'Validation Loss' in line
	])
	# Create values for horizontal axis
	n_epochs = int([lin for lin in lines if 'Epoch(val)' in lin][-1].split('[')[1].split(']')[0])
	train_iters = np.linspace(0, n_epochs, len(train_losses))
	valid_iters = np.arange(1, n_epochs + 1)
	# Plot
	plt.figure()
	sns.set(style="whitegrid")
	plot = sns.lineplot(x=train_iters, y=train_losses, label='Train', linestyle='-')
	sns.lineplot(x=valid_iters, y=valid_losses, label='Valid', marker='o', linestyle='--', color='green')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	fig = plot.figure
	plt.savefig(f'{path}losses.png', bbox_inches='tight')
	
	return fig

def plot_metrics(log_file, path):
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
	# Create values for horizontal axis
	n_epochs = int([lin for lin in lines if 'Epoch(val)' in lin][-1].split('[')[1].split(']')[0])
	valid_iters = np.arange(1, n_epochs + 1)
	# Plot
	plt.figure()
	sns.set(style="whitegrid")
	for i in np.arange(len(metrics_names)):
		plot = sns.lineplot(x=valid_iters, y=metrics_values[:,i], label=metrics_names[i], linestyle='-')
	plt.xlabel('Epochs')
	plt.ylabel('Metrics')
	sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), frameon=False)
	fig = plot.figure
	plt.savefig(f'{path}metrics.png', bbox_inches='tight')
	return fig