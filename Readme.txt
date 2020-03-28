Command line example to execute code:

python3 /media/zaid/zaid1/MAIN.py    --use_gpu 1    --source_dataset Market     --target_dataset CUHK03    --rank 5    --learning_rate 0.003    --dist_metric L1    --model_dir /media/zaid/zaid1/model_    --model_name your_model    --w_loss_rec 0.1    --w_loss_dif 0.1    --w_loss_mmd 0.1    --w_loss_ctr 0.1    --batch_size 8 --pretrain_model_name your_model/model1

For Saliency maps Main file is reid_main.py
For passing original images MAIN file is reid_main_sal_wo.py


For adding any new dataset put the images in /media/zaid/dataset/dataset/YOUR_DATASET

The csv Files must be in : /media/zaid/dataset/dataset/YOUR_DATASET/data/csv/YOUR_DATASET

Modify the Config file for the path to your CSV files or to your Dataset 

Saved models in : /media/zaid/zaid1/model_/YourModel

Saved logs in : /media/zaid/zaid1/log/YourModel

Command line for running tensor board : tensorboard --logdir='/media/zaid/zaid1/log' --bind_all


PLAY_WITH_CSV file : python scripts to create csv files from dataset folders to be used with code, modify and get additional information like cam_id from csv files, make a split of train dataset into train and validation 


The csv files for Market has the train, gallery and query of the original set, Train and Validation(query and gallery) by a 65-10-25 split on train set, Train, query and gallery with and without the camera id column.


Any small changes for testing can be done in the following three files:
reid_main_temp, reid_network_tmp, reid_evaluate_temp. Rest files like config, or reid_dataset or reid_loss is common to all.

