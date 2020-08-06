nohup python my_train_model.py \
--model_type 'InceptionV3' \
--epoch_num 10 \
--X_train 'X_train.npy' \
--Y_train 'Y_train.npy' \
--X_test 'X_test.npy' \
--Y_test 'Y_test.npy' \
--save_model './model' \
--gpu 0 > my_train_model.py.log 
