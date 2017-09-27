from model import Doc2VecModel
import os
import gensim

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

model = Doc2VecModel(lee_train_file,lee_test_file,seed=12345)
model.train_model()

