import torch
import mmcv
import mmdet
import mmengine
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2 # OpenCV is often used by mmcv/mmdet
import math # For calculating grid size

# Import necessary components from MMEngine and MMDetection
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS # Use DATASETS registry from mmdet
from mmcv.visualization import imshow_det_bboxes
from mmdet.structures.det_data_sample import DetDataSample # Import for type checking

# --------------------------------------------------- DATASET VISUALIZATION 

# --- Visualization Function (Modified to draw on a specific axis) ---
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

# --------------------------------------------------- PLOTS 
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
    n_epochs = int([l for l in lines if 'Epoch(val)' in l][-1].split('[')[1].split(']')[0])
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
        [metric for metric in line.split('    ')[1].split('  ') if metric.startswith('coco/')]
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
    n_epochs = int([l for l in lines if 'Epoch(val)' in l][-1].split('[')[1].split(']')[0])
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

# --------------------------------------------------- UTILS 

def get_metrics(log_file):
    # Read log
    with open(log_file) as f:
        lines = f.readlines()
    # Select validation lines
    filtered_lines = [
        [metric for metric in line.split('    ')[1].split('  ') if metric.startswith('coco/')]
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
    best_epoch = int([l for l in lines if 'best' in l][-1].split('_')[-1].split('.')[0])
    metrics = {k: v.item() for k,v in zip(metrics_names, metrics_values[best_epoch-1])}
    return metrics

# --------------------------------------------------- HOOKS

import torch
import torch.distributed as dist
from typing import Optional, Dict, List, Union, Literal

from mmengine.hooks import Hook
from mmengine.model import BaseModule, is_model_wrapper
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from mmengine.dist import get_dist_info, all_reduce_dict
from mmengine.structures import InstanceData

from mmdet.registry import HOOKS
from mmdet.models.detectors.base import BaseDetector

@HOOKS.register_module()
class ValidationLossHook(Hook):
    """
    Calculates and logs validation loss, adaptable to different models.

    This hook iterates through the validation dataloader after standard evaluation.
    It calculates loss based on configured method and mode, aggregates it,
    and logs the average. It can handle models returning a specific loss key
    or requiring summation of all components.

    Args:
        loss_name (str): The target key for the specific total loss in the loss
            dictionary (e.g., 'loss'). If `sum_all_components` is True, this
            name is only used for logging. Defaults to 'loss'.
        calculation_method (Literal['auto', 'forward', 'loss']): Method to use:
            - 'forward': Use `model.forward(..., mode='loss')`. Common for simpler models.
            - 'loss': Use `model.loss()`. Common for complex models (like DETR/DINO).
            - 'auto': Try 'forward' first. If it fails (TypeError/NotImplementedError
              or doesn't return `loss_name`), fall back to 'loss'.
            Defaults to 'auto'.
        force_train_mode (bool): If True, forces `model.train()` during loss
            calculation. Necessary for some models (like DINO) even within
            `torch.no_grad()`. Defaults to False (uses `model.eval()`).
        sum_all_components (bool): If True, ignores `loss_name` for selection
            and sums all valid scalar loss values in the dictionary. Useful
            if the model doesn't return a single total loss key.
            Defaults to False.
        interval (int): Interval of epochs between validation loss calculation.
            Defaults to 1.
        log_prefix (str): Prefix for the logged value key in message hub
            (e.g., 'val'). Defaults to 'val'.
    """
    priority = 'LOW'

    def __init__(self,
                 loss_name: str = 'loss',
                 calculation_method: Literal['auto', 'forward', 'loss'] = 'auto',
                 force_train_mode: bool = False,
                 sum_all_components: bool = False,
                 interval: int = 1,
                 log_prefix: str = 'val'):

        if calculation_method not in ['auto', 'forward', 'loss']:
            raise ValueError("calculation_method must be 'auto', 'forward', or 'loss'")
        if not isinstance(loss_name, str):
            raise TypeError(f'loss_name must be a string, but got {type(loss_name)}')
        if not isinstance(force_train_mode, bool):
             raise TypeError(f'force_train_mode must be a boolean, but got {type(force_train_mode)}')
        if not isinstance(sum_all_components, bool):
             raise TypeError(f'sum_all_components must be a boolean, but got {type(sum_all_components)}')
        if not isinstance(interval, int) or interval <= 0:
             raise ValueError(f'interval must be a positive integer, but got {interval}')
        if not isinstance(log_prefix, str):
             raise TypeError(f'log_prefix must be a string, but got {type(log_prefix)}')

        self.loss_name = loss_name
        self.calculation_method = calculation_method
        self.force_train_mode = force_train_mode
        self.sum_all_components = sum_all_components
        self.interval = interval
        self.log_prefix = log_prefix
        # Internal state for 'auto' mode decision, reset each epoch
        self._auto_method_decision = None # Can be 'forward' or 'loss' after first batch

    def _get_model(self, runner: Runner) -> BaseModule: # Keep BaseModule for generality
        """Gets the unwrapped model."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def _parse_specific_loss(self, losses: dict, device: torch.device, logger: MMLogger) -> Optional[torch.Tensor]:
        """Extracts and sums the specific loss component specified by self.loss_name."""
        if self.loss_name not in losses:
            # Log only once per validation run per strategy attempt
            log_key = f'_logged_missing_{self.loss_name}_warning'
            if not getattr(self, log_key, False):
                logger.warning(
                    f'Specific loss key "{self.loss_name}" not found in output keys: '
                    f'{list(losses.keys())}. Check config or enable sum_all_components. '
                    'Warning shown once per epoch.'
                )
                setattr(self, log_key, True) # Mark as logged for this epoch
            return None

        loss_value = losses[self.loss_name]
        scalar_loss = None

        if isinstance(loss_value, torch.Tensor):
            scalar_loss = torch.sum(loss_value) # Sum elements if not scalar
        elif isinstance(loss_value, (list, tuple)):
            valid = [_l for _l in loss_value if isinstance(_l, torch.Tensor) and _l.numel() > 0]
            if valid: scalar_loss = sum(torch.sum(_l) for _l in valid)
        elif isinstance(loss_value, (int, float)):
             scalar_loss = torch.tensor(loss_value, device=device)

        if scalar_loss is not None:
             if scalar_loss.device != device: scalar_loss = scalar_loss.to(device)
             return scalar_loss.detach()
        else:
             # Log if parsing failed for the specific key unexpectedly
             logger.warning(f"Could not parse loss value for key '{self.loss_name}': {loss_value}")
             return None

    def _calculate_total_loss(self, losses: dict, device: torch.device, logger: MMLogger) -> Optional[torch.Tensor]:
        """Sums all valid scalar loss values found in the losses dictionary."""
        batch_total_loss = torch.tensor(0.0, device=device)
        found_losses = False

        for key, loss_value in losses.items():
            scalar_loss = None
            # Basic check to avoid summing non-loss tensors if possible (e.g., accuracy)
            # This check is heuristic - might need adjustment based on model outputs.
            # We primarily rely on the loss dict *only* containing losses.
            if 'loss' not in key.lower() and 'cost' not in key.lower():
                # logger.debug(f"Skipping potential non-loss key '{key}' during summation.")
                pass # Keep simple for now, assume all tensors are loss components

            if isinstance(loss_value, torch.Tensor):
                 if loss_value.numel() > 0: scalar_loss = torch.sum(loss_value)
            elif isinstance(loss_value, (list, tuple)):
                 valid = [_l for _l in loss_value if isinstance(_l, torch.Tensor) and _l.numel() > 0]
                 if valid: scalar_loss = sum(torch.sum(_l) for _l in valid)

            if scalar_loss is not None:
                if scalar_loss.device != device: scalar_loss = scalar_loss.to(device)
                batch_total_loss += scalar_loss.detach()
                found_losses = True

        if not found_losses:
             log_key = '_logged_no_losses_found_warning'
             if not getattr(self, log_key, False):
                 logger.warning(f"No valid loss tensors found for summation in dict: {list(losses.keys())}. Warning shown once per epoch.")
                 setattr(self, log_key, True)
             return None
        return batch_total_loss

    @torch.no_grad()
    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict] = None) -> None:
        """Calculates validation loss based on configuration."""
        if not self.every_n_epochs(runner, self.interval): return

        # Reset warnings flags for this epoch
        setattr(self, f'_logged_missing_{self.loss_name}_warning', False)
        setattr(self, '_logged_no_losses_found_warning', False)
        self._auto_method_decision = None # Reset auto decision

        model = self._get_model(runner)
        val_dataloader = runner.val_dataloader
        logger: MMLogger = runner.logger
        rank, world_size = get_dist_info()

        try:
            device = getattr(model.data_preprocessor, 'device', next(model.parameters()).device)
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Error determining device: {e}. Aborting.")
            return

        original_mode_is_training = model.training
        current_method = self.calculation_method # Start with configured method

        # Determine initial mode
        use_train_mode = self.force_train_mode
        if current_method == 'auto' and not use_train_mode:
             # In auto mode, we initially try 'forward' in eval mode
             pass # Keep use_train_mode = False for first attempt
        elif current_method == 'loss' and not use_train_mode:
             # If method is 'loss' but train mode isn't forced, still use eval initially
             pass # Keep use_train_mode = False
        # Else use_train_mode respects self.force_train_mode

        total_loss_sum = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(0, device=device)
        processed_batches = 0 # Track batches processed in the loop

        log_loss_name = self.loss_name # Name used in logging messages/keys

        try:
            # Set initial model mode
            if use_train_mode: model.train()
            else: model.eval()

            if rank == 0: logger.info(f'Calculating validation loss ({log_loss_name})...')

            for data_batch in val_dataloader:
                processed_batches += 1
                processed_data = model.data_preprocessor(data_batch, training=False)
                inputs = processed_data['inputs']
                data_samples = processed_data['data_samples']
                losses_dict = None
                batch_loss = None
                attempted_fallback = False

                # --- Determine method for this batch (handles 'auto' mode) ---
                if current_method == 'auto' and self._auto_method_decision is None:
                     # First batch in auto mode: Try 'forward' first
                     effective_method = 'forward'
                     if rank == 0 and processed_batches == 1: logger.info("Auto mode: Trying 'forward' method first.")
                elif self._auto_method_decision:
                     # Auto mode decision made in previous batch
                     effective_method = self._auto_method_decision
                else:
                     # Fixed method ('forward' or 'loss') or decided 'auto'
                     effective_method = current_method

                # --- Attempt selected method ---
                try:
                    current_use_train_mode = use_train_mode # Base mode
                    # If auto decided 'loss', and force_train_mode is false, potentially override?
                    # For now, keep it simple: force_train_mode dictates train() unless eval() needed for forward.
                    # DINO requires train() for loss(), FasterRCNN uses eval() for forward(mode='loss').
                    if effective_method == 'forward' and current_use_train_mode:
                        # If trying 'forward', ensure eval mode unless forced globally (unlikely)
                        if not self.force_train_mode: model.eval()
                        else: model.train() # Respect global force
                    elif effective_method == 'loss':
                         # If trying 'loss', respect force_train_mode
                         if self.force_train_mode: model.train()
                         else: model.eval() # If not forced, try loss() in eval

                    # Call the chosen method
                    if effective_method == 'forward':
                         losses_dict = model.forward(inputs, data_samples, mode='loss')
                    else: # 'loss'
                         losses_dict = model.loss(inputs, data_samples)

                    # Make auto decision persistent for the epoch after first success/failure
                    if current_method == 'auto' and self._auto_method_decision is None:
                         self._auto_method_decision = effective_method # Lock decision

                except (TypeError, NotImplementedError) as e_method:
                     # Method failed (e.g., forward doesn't support mode='loss', or loss() needs train args)
                     if current_method == 'auto' and effective_method == 'forward':
                         # Auto mode: 'forward' failed, try 'loss' method instead
                         if rank == 0 and not attempted_fallback: logger.warning(f"'forward(mode='loss')' failed ({e_method}), Auto mode: Falling back to 'loss' method.")
                         effective_method = 'loss'
                         self._auto_method_decision = 'loss' # Lock decision
                         attempted_fallback = True

                         # Retry with 'loss' method (respecting force_train_mode)
                         try:
                            if self.force_train_mode: model.train()
                            else: model.eval() # Try loss() in eval if not forced
                            losses_dict = model.loss(inputs, data_samples)
                         except Exception as e_fallback:
                             logger.error(f"[Rank {rank}] Fallback method 'loss()' also failed: {e_fallback}", exc_info=True)
                             raise # Re-raise the exception from the fallback attempt
                     else:
                         # Fixed method failed, or fallback failed
                         logger.error(f"[Rank {rank}] Method '{effective_method}' failed: {e_method}", exc_info=True)
                         raise # Re-raise the original error

                # --- Parse the result ---
                if losses_dict is None:
                     # Should not happen if exceptions are handled, but safeguard
                     logger.error(f"[Rank {rank}] losses_dict is None after calculation attempt, skipping batch.")
                     continue

                if self.sum_all_components:
                     batch_loss = self._calculate_total_loss(losses_dict, device, logger)
                     log_loss_name = self.loss_name + " (summed)" if self.loss_name else "total_loss (summed)"
                else:
                     batch_loss = self._parse_specific_loss(losses_dict, device, logger)
                     # If specific loss not found, maybe we should try summing anyway? Add option?
                     # For now, if specific key not found, batch_loss is None.
                     log_loss_name = self.loss_name # Use specified name

                # Accumulate
                if batch_loss is not None:
                    total_loss_sum += batch_loss # Already detached
                    num_batches += 1

        except Exception as e:
            logger.error(f"[Rank {rank}] Unhandled error during validation loss calculation loop: {e}", exc_info=True)
            total_loss_sum = None # Mark as failed
        finally:
            # Restore original model mode
            if original_mode_is_training: model.train()
            else: model.eval()

        if total_loss_sum is None: return # Error already logged

        # Aggregate results across all GPUs
        if world_size > 1:
            stats_to_reduce = {'val_loss_sum': total_loss_sum, 'val_batches_sum': num_batches}
            reduced_stats = all_reduce_dict(stats_to_reduce, op='sum')
            total_loss_sum = reduced_stats['val_loss_sum']
            num_batches = reduced_stats['val_batches_sum']

        # Calculate and log average loss (only on rank 0)
        if rank == 0:
            if num_batches.item() > 0:
                average_loss = (total_loss_sum / num_batches).item()
                # Use the name determined during parsing (specific or summed)
                final_log_loss_name = log_loss_name if batch_loss is not None else self.loss_name # Fallback name
                log_str = f'Validation Loss ({final_log_loss_name}): {average_loss:.4f}'
                logger.info(log_str)
                # Use a consistent key format, potentially indicating summation if it happened
                log_key_suffix = "_summed" if self.sum_all_components or (batch_loss is not None and log_loss_name.endswith("(summed)")) else ""
                log_key = f'{self.log_prefix}/{self.loss_name}{log_key_suffix}'
                runner.message_hub.update_scalar(log_key, average_loss)
            elif processed_batches > 0 : # Processed batches but got 0 valid losses
                 logger.warning(
                    f'Validation loss ({self.loss_name}) computed as 0 or loss key/components consistently not found '
                    f'across {processed_batches} batches.'
                )
            # else: Dataloader was empty, do nothing.


# --------------------------------------------------- RESULTS

from mmdet.evaluation import bbox_overlaps
import numpy as np
from mmengine.utils import ProgressBar

def collect_detections_and_ground_truths(dataset,
                                         results,
                                         score_thr=0.0,
                                         tp_iou_thr=0.5,
                                         max_dets=300):
    """Collect predicted and ground truth classes for each image.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[dict]): A list of detection results per image.
        score_thr (float): Score threshold to filter bboxes.
        tp_iou_thr (float): IoU threshold to be considered a TP.
        max_dets (int): Max detections per image (usually 100 for COCO mAP).

    Returns:
        tuple: (list of predicted classes, list of ground truth classes)
    """
    all_pred_classes = []
    all_gt_classes = []

    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))

    for idx, per_img_res in enumerate(results):
        result = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']

        gt_bboxes = np.array([gt['bbox'] for gt in gts], dtype=np.float32)
        gt_labels = np.array([gt['bbox_label'] for gt in gts], dtype=np.int64)

        det_bboxes = result['bboxes'].numpy()
        det_scores = result['scores'].numpy()
        det_labels = result['labels'].numpy()

        # Filter by score threshold
        keep = det_scores >= score_thr
        det_bboxes = det_bboxes[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]

        # Keep top-k detections (COCO default: 100)
        if len(det_scores) > max_dets:
            topk_inds = np.argsort(det_scores)[::-1][:max_dets]
            det_bboxes = det_bboxes[topk_inds]
            det_scores = det_scores[topk_inds]
            det_labels = det_labels[topk_inds]

        # No predictions or GTs
        if len(gt_bboxes) == 0 and len(det_bboxes) == 0:
            prog_bar.update()
            continue
        elif len(gt_bboxes) == 0:
            # All predictions are false positives
            all_pred_classes.extend(det_labels.tolist())
            all_gt_classes.extend([9999] * len(det_labels))
            prog_bar.update()
            continue
        elif len(det_bboxes) == 0:
            # All GTs are false negatives
            all_pred_classes.extend([9999] * len(gt_labels))
            all_gt_classes.extend(gt_labels.tolist())
            prog_bar.update()
            continue

        # IoU matrix
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        matched_gts = set()
        matched_preds = set()

        for i in range(len(det_bboxes)):
            best_iou = 0
            best_gt_idx = -1
            for j in range(len(gt_bboxes)):
                if j in matched_gts:
                    continue
                if ious[i, j] >= tp_iou_thr and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt_idx = j
            if best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                matched_preds.add(i)
                all_pred_classes.append(det_labels[i])
                all_gt_classes.append(gt_labels[best_gt_idx])

        # False positives (preds not matched)
        for i in range(len(det_bboxes)):
            if i not in matched_preds:
                all_pred_classes.append(det_labels[i])
                all_gt_classes.append(9999)

        # False negatives (GTs not matched)
        for j in range(len(gt_bboxes)):
            if j not in matched_gts:
                all_pred_classes.append(9999)
                all_gt_classes.append(gt_labels[j])

        prog_bar.update()

    return all_pred_classes, all_gt_classes