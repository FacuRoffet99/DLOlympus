import torch
from typing import Optional, Dict, Literal

from mmengine.hooks import Hook
from mmengine.model import BaseModule, is_model_wrapper
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from mmengine.dist import get_dist_info, all_reduce_dict

from mmdet.registry import HOOKS


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
			if valid: 
				scalar_loss = sum(torch.sum(_l) for _l in valid)
		elif isinstance(loss_value, (int, float)):
			scalar_loss = torch.tensor(loss_value, device=device)

		if scalar_loss is not None:
			if scalar_loss.device != device: 
				scalar_loss = scalar_loss.to(device)
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
				if loss_value.numel() > 0: 
					scalar_loss = torch.sum(loss_value)
			elif isinstance(loss_value, (list, tuple)):
				valid = [_l for _l in loss_value if isinstance(_l, torch.Tensor) and _l.numel() > 0]
				if valid: 
					scalar_loss = sum(torch.sum(_l) for _l in valid)

			if scalar_loss is not None:
				if scalar_loss.device != device: 
					scalar_loss = scalar_loss.to(device)
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
		if not self.every_n_epochs(runner, self.interval): 
			return

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
			if use_train_mode: 
				model.train()
			else: 
				model.eval()

			if rank == 0: 
				logger.info(f'Calculating validation loss ({log_loss_name})...')

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
					if rank == 0 and processed_batches == 1: 
						logger.info("Auto mode: Trying 'forward' method first.")
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
						if not self.force_train_mode: 
							model.eval()
						else: 
							model.train() # Respect global force
					elif effective_method == 'loss':
						# If trying 'loss', respect force_train_mode
						if self.force_train_mode: 
							model.train()
						else: 
							model.eval() # If not forced, try loss() in eval

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
						if rank == 0 and not attempted_fallback: 
							logger.warning(f"'forward(mode='loss')' failed ({e_method}), Auto mode: Falling back to 'loss' method.")
						effective_method = 'loss'
						self._auto_method_decision = 'loss' # Lock decision
						attempted_fallback = True

						 # Retry with 'loss' method (respecting force_train_mode)
						try:
							if self.force_train_mode: 
								model.train()
							else: 
								model.eval() # Try loss() in eval if not forced
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
			if original_mode_is_training: 
				model.train()
			else: 
				model.eval()

		if total_loss_sum is None: 
			return # Error already logged

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