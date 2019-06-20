from CNNModels import *
from dataLoader import *
from keras.callbacks import TensorBoard
from keras import metrics
from keras.models import load_model


# CLASS_WEIGHTS = { 0 : 0.1,
#                  1 : 1, 
#                  2 : 1 ,
#                  3 : 1, 
#                  4 : 1, 
#                  5 : 1 , 
#                  6 : 1, 
#                  7 : 1,
#                  8 : 1 , 
#                  9 : 1 , 
#                  10 : 1, 
#                  11 : 1}

CLASS_WEIGHTS = [1 , 1. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,1.]
def class_weighted_pixelwise_crossentropy(target, output, weights=CLASS_WEIGHTS):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    #weights = [0.8, 0.2]
    return -tf.reduce_sum(tf.reduce_mean(target * weights * tf.log(output), axis=[0,1]))

def weighted_categorical_crossentropy(target, output, from_logits=False, axis=-1,weights=CLASS_WEIGHTS):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    _EPSILON = 1e-7
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * weights * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

if __name__=="__main__":    
    data_gen_args_ims = dict(#samplewise_center=True,
                    rescale=1./255.,
                    rotation_range=20,
                    width_shift_range=0.20,
                    height_shift_range=0.20,
                    shear_range=0.20,
                    zoom_range=0.20,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    #validation_split=0.1)
    data_gen_args_mask = dict(#samplewise_center=True,
                    #rescale=1. / 255,
                    rotation_range=20,
                    width_shift_range=0.20,
                    height_shift_range=0.20,
                    shear_range=0.20,
                    zoom_range=0.20,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    #validation_split=0.1)
    

    trainingData = generateTrainValDataGenerator(10,'train', 'im1','im2','gt',data_gen_args_ims, data_gen_args_mask, save_to_dir = None, target_size = (224,224))
    model = changeNet_VGG2()


    pretrained_model = VGG16(include_top=True, weights='imagenet')
    pretrained_model.summary()
    DICT_MAPPING = {'conv2d_1' : 'block1_conv1' , 
                    'conv2d_2' : 'block1_conv2',
                    'conv2d_3' : 'block2_conv1' ,
                    'conv2d_4' : 'block2_conv2' ,
                    'conv2d_5' : 'block3_conv1' ,
                    'conv2d_6' : 'block3_conv2' ,
                    'conv2d_7' : 'block3_conv3' ,
                    'conv2d_8' : 'block4_conv1' ,
                    'conv2d_9' : 'block4_conv2' ,
                    'conv2d_10' : 'block4_conv3' ,
                    'conv2d_11' : 'block5_conv1' ,
                    'conv2d_12' : 'block5_conv2' ,
                    'conv2d_13' : 'block5_conv3' }
                    #'dense' : 'fc1',
                    #'dense_1' : 'fc2'}
    # Instanitate the model weights by calling it on a dummy tensor
    model( [tf.convert_to_tensor(np.zeros([1,224,224,3],dtype=np.float64)), tf.convert_to_tensor(np.zeros([1,224,224,3],dtype=np.float64))] )
    for key, value in DICT_MAPPING.items():
        print(key)
        print(value)
        model.get_layer(key).set_weights(pretrained_model.get_layer(value).get_weights())
        model.get_layer(key).trainable=False
    model.summary()
    #trainingData, valData = generateTrainValDataGenerator(10,'train', 'im1','im2','gt',data_gen_args_ims, data_gen_args_mask, save_to_dir = None)
    #trainingData = trainvalDataGenerator(trainingData)
    #valData = trainvalDataGenerator(valData)

    #model = changeNet()
    #model.compile(optimizer = Adam(lr = 1e-4), loss = weighted_categorical_crossentropy, metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 5e-5), loss = class_weighted_pixelwise_crossentropy, metrics = ['accuracy', metrics.categorical_accuracy])
    
    model.load_weights("pretrainedVGGChange.hdf5")
    
    #model_checkpoint = ModelCheckpoint('pretrainedVGGChange.hdf5', monitor='loss',verbose=1, save_best_only=True)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                       write_graph=True, write_images=True)
    # model.fit_generator(trainingData, steps_per_epoch=150,epochs=100,callbacks=[model_checkpoint, tensorboard])

    data_gen_args_ims_test = dict(#samplewise_center=True,
                    rescale=1./255.,
                    fill_mode='nearest')
    data_gen_args_mask_test = dict(fill_mode='nearest')

    testingEvalData = generateTestGenerator(1,'test', 'im1','im2','gt',data_gen_args_ims_test, data_gen_args_mask_test, save_to_dir = None, target_size = (224,224))
    testResults = model.evaluate_generator(testingEvalData, steps=429, verbose=1)
    print(model.metrics_names)
    print(testResults)

    results = model.predict_generator(testingEvalData,steps=429,verbose=1)
    saveResult("output/",results, target_size = (224,224))


    testingAllNoChangeEvalData = generateAllNoChangeTestGenerator(1,'test', 'im1','im2','gt',data_gen_args_ims_test, data_gen_args_mask_test, save_to_dir = None, target_size = (224,224))
    zeromodel = zeroNet()
    zeromodel.compile(optimizer = Adam(lr = 1e-4), loss = class_weighted_pixelwise_crossentropy, metrics = ['accuracy', metrics.categorical_accuracy])
    testResultsAllNoChange = zeromodel.evaluate_generator(testingAllNoChangeEvalData, steps=429, verbose=1)
    print(testResultsAllNoChange)



    #testData = testDataGenerator("test/im1" , "test/im2", "test/gt", target_size = (224,224))
    #results = model.predict_generator(testingEvalData,steps=429,verbose=1)
    #tf.metrics.accuracy()
    #print(results.shape)
    #saveResult("output/",results, target_size = (224,224))

    

    # data_gen_args_ims = dict(#samplewise_center=True,
    #                 rescale=1. / 255,
    #                 rotation_range=0.2,
    #                 width_shift_range=0.05,
    #                 height_shift_range=0.05,
    #                 shear_range=0.05,
    #                 zoom_range=0.05,
    #                 horizontal_flip=True,
    #                 fill_mode='nearest')
    # data_gen_args_mask = dict(#samplewise_center=True,
    #                 #rescale=1. / 255,
    #                 rotation_range=0.2,
    #                 width_shift_range=0.05,
    #                 height_shift_range=0.05,
    #                 shear_range=0.05,
    #                 zoom_range=0.05,
    #                 horizontal_flip=True,
    #                 fill_mode='nearest')