# import pickle
# f_data = open('/gdrive/MyDrive/data_vgg', 'rb')
# unpickler_data = pickle.Unpickler(f_data)

# f_labels = open('/gdrive/MyDrive/labels_vgg', 'rb')
# unpickler_labels = pickle.Unpickler(f_labels)

# import psutil
# import time

# sec = 0
# dataset2 = None
# steps_per_epoch = train_config.num_images // 10240
# for i in range(steps_per_epoch):
#   batch_data = unpickler_data.load()
#   batch_labels = unpickler_labels.load()
#   batch_dataset = tf.data.Dataset.from_tensor_slices((batch_data, batch_labels))
#   if dataset2 is None:
#     dataset2 = batch_dataset
#   dataset2 = dataset2.concatenate(batch_dataset)
#   if (i+1)%1==0:
#     print(i+1,'/',steps_per_epoch , '  --  ', psutil.virtual_memory().percent)
#   # if psutil.virtual_memory().percent > 80 :
#   #   count_old = 0
#   #   old = psutil.virtual_memory().percent
#   #   while count_old < 10 :
#   #     if psutil.virtual_memory().percent == old:
#   #       count_old +=1
#   #     else:
#   #       count_old = 0
#   #     old = psutil.virtual_memory().percent
#   #     time.sleep(1)
#   #     sec+=1
#   #     print('sec -', sec , '-  ', psutil.virtual_memory().percent)

# # build model
# with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
#   model = InceptionResNetV1(num_classes=train_config.num_classes,
#                             embedding_size=train_config.embedding_size,
#                             model_type='ArcHead',
#                             name="InceptionResNetV1")
#   # compile model
#   model.compile(loss=CosfaceLoss(margin=0.5, scale=64, n_classes=train_config.num_classes),
#                 optimizer=Adam(learning_rate=0.001),
#                 metrics=['accuracy'])

# model.fit(
#     training_dataset, 
#     steps_per_epoch=train_config.num_images // batch_size, epochs=100, 
#     # callbacks =[model_checkpoint_callback]
# )
