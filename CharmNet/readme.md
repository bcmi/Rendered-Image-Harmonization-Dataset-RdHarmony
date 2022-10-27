python train.py --name experiment_name --model dastyle --lr 1e-4 --is_train 1 --display_id 0 --gpu_ids 0 --norm batch --preprocess resize_and_crop --batch_size 8 --lr_policy step --lr_decay_iters 4702400

python test.py --name da_test --model dastyle --dataset_mode real --gpu_ids 0 --is_train 0 --preprocess resize --norm batch --epoch 21 --eval


python test_save.py --name da_test --model dastyle --dataset_mode real --gpu_ids 0 --is_train 0 --preprocess resize --norm batch --epoch 21 --eval