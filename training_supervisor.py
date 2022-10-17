from datetime import datetime
import logging
import os.path

import mlflow
import tensorflow as tf
from tqdm import tqdm


class TrainingSupervisor(object):

    def __init__(self, 
                 optimizer, loss_fn,
                 model,
                 save_freq,
                 monitor, mode,
                 training_dir,
                 name,
                 train_dataloader = None, 
                 validation_dataloader = None,
                 datasetQueue = None,
                 ):
        super(TrainingSupervisor, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_freq = save_freq

        self.datasetQueue = datasetQueue
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validation_dataloader
        if self.train_dataloader != None:
          self.datatrain_generator = iter(self.train_dataloader)

        # Setup metrics
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(name='train_accuracy', dtype=tf.float32),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean", dtype=tf.float32)
        }
        self.monitor = self.metrics[monitor]
        self.mode = mode

        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(1, trainable=False, dtype=tf.int64),
            'monitor_value': tf.Variable(0, trainable=False, dtype=tf.float32)
        }


    @tf.function
    def _train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            regularization_loss = self.model.losses
            predict_loss = self.loss_fn(y_batch, logits)
            total_loss = predict_loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return logits, total_loss

    @tf.function
    def _update_metrics(self, **kwargs):
        """

        :param kwargs: some parameter we can change for future
        :return: None (Update state of metrics)
        """
        # get parameter
        labels = kwargs['labels']
        logits = kwargs['logits']
        loss = kwargs['loss']

        self.metrics['categorical_accuracy'].update_state(labels, logits)
        self.metrics['loss'].update_state(loss)

    def _reset_metrics(self):
        for key, metric in self.metrics.items():
            metric.reset_states()

    def _checkpoint(self):
        def _check_value(v1, v2, mode):
            if (v1 < v2) & (mode == 'min'):
                return True
            elif (v1 > v2) & (mode == 'max'):
                return True
            else:
                return False

        # Get previous and current monitor values.
        previous = self.schedule['monitor_value'].numpy()
        current = self.monitor.result()

        if previous == 0.0:
            self.schedule['monitor_value'].assign(current)

        if _check_value(current, previous, self.mode):
            print("Monitor value improved from {:.4f} to {:.4f}.".format(previous, current))

            # Update the schedule.
            self.schedule['monitor_value'].assign(current)

            # And save the model.
            best_model_path = self.scout.save()
            print("Best model found and saved: {}".format(best_model_path))
        else:
            print("Monitor value not improved: {:.4f}, latest: {:.4f}.".format(previous, current))

        ckpt_path = self.manager.save()
        self._reset_metrics()
        print(f"Checkpoint saved at global step {self.schedule['step']}, to file: {ckpt_path}")


    def train(self, epochs, steps_per_epoch):
        initial_epoch = 1
        global_step = 0
        initial_step = global_step % steps_per_epoch
        print("Resume training from global step: {}, epoch: {}".format(global_step, initial_epoch))
        print("Current step is: {}".format(initial_step))

        for epoch in range(initial_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                                ascii="->", colour='#1cd41c')

            for index in range(initial_step, steps_per_epoch):
                if self.datasetQueue != None:
                  data, labels = self.datasetQueue.getBatch()
                else:
                  data, labels = next(self.datatrain_generator)

                logits, loss = self._train_step(x_batch=data, y_batch=labels)

                # update metrics
                self._update_metrics(labels=labels, logits=logits, loss=loss)

                # update training schedule
                self.schedule['step'].assign_add(1)

                # update process bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": "{:.4f}".format(loss.numpy()[0]),
                    "accuracy": "{:.4f}".format(self.metrics['categorical_accuracy'].result().numpy())
                })

                # Log and checkpoint the model.
                # if int(self.schedule['step']) % self.save_freq == 0:
                #     self._log_to_tensorboard()
                #     self._checkpoint()
                    # mình sẽ gọi evaluate ở đây.

            # Mlflow logging
            # self._mlflow_log()

            # Save the last checkpoint.
            # self._log_to_tensorboard()
            # self._checkpoint()

            # update epoch
            self.schedule['epoch'].assign_add(1)

            # reset iterate
            if self.datatrain_generator != None:
              self.datatrain_generator = iter(self.train_dataloader)

            # clean up process bar
            progress_bar.close()
            initial_step = 0


    def override_schedule(self):
        raise NotImplementedError

