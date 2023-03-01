## Example train
python main.py ctdet --dataset copilot --batch_size 1 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch smallhourglass

## Example test
python test.py ctdet --dataset copilot --keep_res --load_model ../exp/copilot/ctdet/default/model_last.pth --arch smallhourglass