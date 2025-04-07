# train
python -u run.py --task_name anomaly_detection --anomaly_ratio 0.5 --is_training 1 --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 3 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# # test only
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode test --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# Train Adv model
# # Precision
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --adv_epoch 1 --mode train_adv_gen  --attack_method Transformer_Adv --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.5


# Gen-Adversarial Examples
# # Precision
# # # Random:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode gen_adv_test --attack_method Random --attack_target Precision  --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode gen_adv_test --attack_method FGSM --attack_target Precision  --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02

# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode gen_adv_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02

# # Test-Adversarial Examples
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode test_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False 

# Visualization
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode visualize_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# eval
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode eval_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# # Adversarial Evaluation
# # # FGSM
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode eval_adv --attack_method FGSM --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode eval_adv --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# Gen_Adv_Train
# # Precision
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode gen_adv_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --epsilon 0.02

# Adversarial Training
# # Precision
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode defence_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode defence_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug False


# Adversarial Defense
# # Precision
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode defence_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 0.5 --mode defence_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SWaT/ --model_id SWAT --model ModernTCN --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0003 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug False
