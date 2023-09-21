## Example train
python main.py ctdet --dataset copilot --batch_size 1 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch smallhourglass

python main.py ctdet --dataset copilot --arch hourglass --batch_size 2 --master_batch 4 --lr 2.5e-4 --gpus 0


python main.py polydet --val_intervals 12 --exp_id from_coco_16_resnetdcn101_l1_no_fg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset copilot --arch hourglass  --batch_size 1 --lr 2e-4 --gpus 0

python main.py polydet --val_intervals 12 --exp_id from_copilot_16_resnetdcn101_l1_no_fg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset copilot --arch hourglass  --batch_size 1 --lr 2e-4 --gpus 0 --not_rand_crop

python main.py polydet --val_intervals 5 --exp_id prueba2 --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset copilot --arch hourglass  --batch_size 1 --lr 2e-4 --gpus 1

python main.py polydet --exp_id centerPoly_train --elliptical_gt  --depth_weight 0.1 --poly_weight 1 --nbr_points 16 --dataset copilot --arch hourglass  --batch_size 2 --master_batch 2 --lr 2e-4 --trainval --num_workers=4

python main.py polydet --exp_id centerPoly_train2 --elliptical_gt  --depth_weight 0.1 --poly_weight 1 --nbr_points 16 --dataset copilot --arch smallhourglass  --batch_size 2 --master_batch 2 --lr 2e-4 --trainval --num_workers=4

## Example test
python test.py ctdet --dataset copilot --keep_res --load_model ../exp/copilot/ctdet/default/model_last.pth --arch smallhourglass --not_prefetch_test

python test.py polydet --exp_id epochs_500 --nbr_points 16 --dataset copilot --arch smallhourglass --load_model ../exp/copilot/polydet/epochs_500/model_last.pth --not_prefetch_test

python test.py polydet --exp_id centerPoly_train2 --nbr_points 16 --dataset copilot --arch smallhourglass --load_model ../exp/copilot/polydet/centerPoly_train2/model_last.pth --not_prefetch_test --trainval
