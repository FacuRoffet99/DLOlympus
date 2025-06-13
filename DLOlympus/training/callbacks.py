
from fastai.callback.tracker import TrackerCallback
from fastcore.basics import store_attr

# Same as FastAI's callback, but with 'weights_only=False' to comply with newer torch versions.
class SaveModelCallback(TrackerCallback):
	"A `TrackerCallback` that saves the model's best during training and loads it at the end."
	order = TrackerCallback.order+1
	def __init__(self, 
		monitor='valid_loss', # value (usually loss or metric) being monitored.
		comp=None, # numpy comparison operator; np.less if monitor is loss, np.greater if monitor is metric.
		min_delta=0., # minimum delta between the last monitor value and the best monitor value.
		fname='model', # model name to be used when saving model.
		every_epoch=False, # if true, save model after every epoch; else save only when model is better than existing best.
		at_end=False, # if true, save model when training ends; else load best model if there is only one saved model.
		with_opt=False, # if true, save optimizer state (if any available) when saving model. 
		reset_on_fit=True # before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
    ):
		super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
		assert not (every_epoch and at_end), "every_epoch and at_end cannot both be set to True"
		# keep track of file path for loggers
		self.last_saved_path = None
		store_attr('fname,every_epoch,at_end,with_opt')

	def _save(self, name): self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

	def after_epoch(self):
		"Compare the value monitored to its best score and save if best."
		if self.every_epoch:
			if (self.epoch%self.every_epoch) == 0: self._save(f'{self.fname}_{self.epoch}')
		else: #every improvement
			super().after_epoch()
			if self.new_best:
				print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
				self._save(f'{self.fname}')

	def after_fit(self, **kwargs):
		"Load the best model."
		if self.at_end: self._save(f'{self.fname}')
		elif not self.every_epoch: self.learn.load(f'{self.fname}', with_opt=self.with_opt, weights_only=False)