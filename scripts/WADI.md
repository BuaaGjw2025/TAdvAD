# train
python -u run.py --task_name anomaly_detection --anomaly_ratio 0.1  --is_training 1 --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0001 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

### get threshold
# # test only
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode test --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# # test second time
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --mode test_ori --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# # Train Adv model
# Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --adv_epoch 1 --mode train_adv_gen --attack_method Transformer_Adv  --anomaly_ratio 0.1 --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.5


# Gen-Adversarial Examples
# # Precision 
# # # Random:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --attack_method Random  --attack_target Precision  --mode gen_adv_test --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method FGSM  --attack_target Precision  --mode gen_adv_test --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.02
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --mode gen_adv_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.02


# Visualization
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --mode visualize_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# eval_adv
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --mode eval_adv --attack_method FGSM --attack_target Precision --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --mode eval_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# Gen_Adv_Train
# # Precision
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method Transformer_Adv  --attack_target Precision --mode gen_adv_train --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02

# Adversarial Training
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method Transformer_Adv  --attack_target Precision  --mode defence_train --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug True 

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method Transformer_Adv  --attack_target Precision  --mode defence_train --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug False


# Adversarial Defense
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method Transformer_Adv  --attack_target Precision  --mode defence_test --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.1 --attack_method Transformer_Adv  --attack_target Precision  --mode defence_test --root_path ./all_datasets/WADI/ --model_id WADI --model ModernTCN --data WADI --seq_len 100 --label_len 0 --pred_len 0 --enc_in 96 --c_out 96 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 127 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug False

