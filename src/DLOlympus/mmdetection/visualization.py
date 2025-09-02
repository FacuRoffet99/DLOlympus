import cv2
import mmcv
import math 
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from mmdet.structures.det_data_sample import DetDataSample
from mmcv.visualization import imshow_det_bboxes


def visualize_sample(dataset_instance, index, class_names_list, norm_cfg, ax):
	"""
	Fetches a sample, processes it, gets the annotated image using imshow_det_bboxes,
	and draws it onto the provided matplotlib axis object 'ax'.
	Returns True on success, False on failure for this sample.
	"""
	try:
		data_item = dataset_instance[index]
	except Exception as e:
		print(f"Error getting item at index {index}: {e}")
		ax.set_title(f"Error loading Index: {index}")
		ax.axis('off')
		return False # Indicate failure

	img_tensor = None
	data_sample = None

	# Simplified extraction logic assuming PackDetInputs format
	if 'inputs' in data_item and 'data_samples' in data_item:
		img_tensor_maybe_list = data_item['inputs']
		data_samples_list = data_item['data_samples']
		if isinstance(img_tensor_maybe_list, list) and img_tensor_maybe_list:
			img_tensor = img_tensor_maybe_list[0]
		elif torch.is_tensor(img_tensor_maybe_list):
			img_tensor = img_tensor_maybe_list
		if isinstance(data_samples_list, list) and data_samples_list:
			if isinstance(data_samples_list[0], DetDataSample):
				data_sample = data_samples_list[0]
		elif isinstance(data_samples_list, DetDataSample):
			data_sample = data_samples_list

	# Handle potential fallback or misconfiguration
	if not torch.is_tensor(img_tensor):
		print(f"Error: Could not obtain a valid image tensor for index {index}.")
		ax.set_title(f"Image tensor error\nIndex: {index}")
		ax.axis('off')
		return False # Indicate failure

	# --- Process Image (Permute, Denormalize, uint8) ---
	img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
	if norm_cfg:
		try:
			mean = np.array(norm_cfg['mean'], dtype=np.float32)
			std = np.array(norm_cfg['std'], dtype=np.float32)
			to_bgr = norm_cfg.get('to_rgb', False)
			img_display = mmcv.imdenormalize(img_display.copy(), mean, std, to_bgr=to_bgr)
		except Exception as e:
			print(f"Warning: Error during imdenormalize for index {index}: {e}. Displaying raw.")
	if img_display.dtype != np.uint8:
		img_min, img_max = img_display.min(), img_display.max()
		if img_max <= 1.1 and img_min >= -0.1 and not norm_cfg: # Allow slightly wider range for float noise
			img_display = (img_display * 255.0)
		img_display = np.clip(img_display, 0, 255).astype(np.uint8)

	# --- Extract Annotations ---
	bboxes = np.zeros((0, 4), dtype=np.float32)
	labels = np.zeros((0,), dtype=np.int64)
	if data_sample is not None and hasattr(data_sample, 'gt_instances') and len(data_sample.gt_instances) > 0:
		bboxes = data_sample.gt_instances.bboxes.cpu().numpy()
		labels = data_sample.gt_instances.labels.cpu().numpy()
	else:
		print("No ground truth instances found or data_sample invalid.")

	# --- Generate Annotated Image using imshow_det_bboxes ---
	returned_image = None
	try:
		img_display_contiguous = np.ascontiguousarray(img_display)
		returned_image = imshow_det_bboxes(
			img_display_contiguous, # BGR image
			bboxes,
			labels,
			class_names=class_names_list,
			show=False, # IMPORTANT: Keep False
			bbox_color='green',
			text_color='white',
			thickness=1,
		)
	except Exception as e:
		print(f"!!! Error during imshow_det_bboxes call for index {index}: {e}")
		ax.set_title(f"Plotting error\nIndex: {index}")
		ax.axis('off')
		return False # Indicate failure

	# --- Display the returned image on the provided axis 'ax' ---
	if returned_image is not None:
		# Convert BGR (from mmcv) to RGB (for matplotlib)
		img_rgb_with_boxes = cv2.cvtColor(returned_image, cv2.COLOR_BGR2RGB)
		ax.imshow(img_rgb_with_boxes)
		title = f"Index: {index}"
		if data_sample and hasattr(data_sample, 'img_id'): # Add img_id if available
			title += f"\nimg_id: {data_sample.img_id}"
		ax.set_title(title, fontsize=10)
		ax.axis('off')
		return True # Indicate success
	else:
		print(f"!!! Error: imshow_det_bboxes returned None for index {index}.")
		ax.set_title(f"Plotting Failed\nIndex: {index}")
		ax.axis('off')
		return False # Indicate failure

def show_samples(cfg, n=9, unique=False):
	init_default_scope(cfg.get('default_scope', 'mmdet'))

	# --- Prepare Dataset ---
	try:
		dataset_cfg = cfg.train_dataloader.dataset
		pipeline_cfg = dataset_cfg.pipeline
	except AttributeError:
		try:
			dataset_cfg = cfg.data.train.dataset
			pipeline_cfg = dataset_cfg.pipeline
		except AttributeError:
			raise ValueError("Could not find dataset configuration in the config file.")
	dataset = DATASETS.build(dataset_cfg)
	if not dataset:
		raise ValueError("The dataset is empty or failed to load.")
	num_total_samples = len(dataset)
	class_names = dataset.metainfo['classes']

	# --- Inspect Pipeline for Normalization ---
	img_norm_cfg_dict = None
	for transform in pipeline_cfg:
		if transform['type'] == 'PackDetInputs':
			break
		if transform['type'] == 'Normalize':
			img_norm_cfg_dict = transform
			break

	if num_total_samples == 0:
		print("Dataset is empty. Cannot visualize.")
	else:
		# Determine grid size (try to make it squarish)
		cols = math.ceil(math.sqrt(n))
		rows = math.ceil(n / cols)

		# Create the figure and axes grid
		fig_width = cols * 5 # Adjust multiplier for desired subplot size
		fig_height = rows * 5
		fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
		# Flatten axes array for easy iteration, handles single row/col cases
		axes = axes.flatten()

		indices_to_process = []
		figure_title = ""

		if unique:
			# Select ONE base index
			if num_total_samples > 0:
				base_index = random.randint(0, num_total_samples - 1)
				indices_to_process = [base_index] * n
				figure_title = f"Multiple Augmentations of Sample Index: {base_index}"
			else:
				print("Dataset empty, cannot run in unique mode.")
		else:
			# Select multiple DIFFERENT indices
			num_to_show = min(n, num_total_samples)
			if num_to_show < n:
				print(f"Warning: Requested {n} samples, but dataset only has {num_total_samples}. Showing {num_to_show}.")
			indices_to_process = random.sample(range(num_total_samples), num_to_show)
			figure_title = f"{len(indices_to_process)} Random Augmented Samples"

		# Loop through the number of plots we want to create
		successful_plots = 0
		for i in range(n):
			if i < len(indices_to_process):
				current_index = indices_to_process[i]
				ax = axes[i] # Get the specific subplot axis
				# Call the updated function to draw on this axis
				success = visualize_sample(
					dataset, current_index, class_names, img_norm_cfg_dict, ax
				)
				if success:
					successful_plots += 1
			else:
				# Hide unused subplots if NUM_SAMPLES_TO_VISUALIZE doesn't fill the grid
				axes[i].axis('off')

		# Add a main title to the whole figure
		fig.suptitle(figure_title, fontsize=16)

		# Adjust layout to prevent titles/labels overlapping
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

		# Show the entire figure with all subplots
		plt.show()