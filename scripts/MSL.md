# Original train
python -u run.py --task_name anomaly_detection --anomaly_ratio 1 --is_training 1 --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL  --seq_len 100 --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --train_epochs 3 

# test only
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode test --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# test second time
python -u run_Adv.py --task_name anomaly_detection --adv_epoch 3 --mode train_adv_gen --attack_method Transformer_Adv --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# Train Adv model
# # attack_target: Precision
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode train_adv_gen  --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.5

# Gen-Adversarial Examples
# # Precision
# # # Random:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_test  --attack_method Random --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.15
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_test  --attack_method FGSM --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.15
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_test  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --cross_attack False --epsilon 0.15


# Visualization
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode visualize_adv  --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False

# Evaluation
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode eval_adv  --attack_method FGSM --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode eval_adv  --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 2 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False


# Gen_Adv_Train
# # Precision
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_train  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --epsilon 0.15

# Adversarial Training
# # Precision:
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_train  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_train  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False  --apply_adv_aug False
# Defence_test
# # Precision
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_test  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_test  --attack_method Transformer_Adv  --attack_target Precision --root_path ./all_datasets/MSL/ --model_id MSL --model ModernTCN --data MSL --enc_in 55 --c_out 55 --ffn_ratio 1 --patch_size 8 --patch_stride 4 --num_blocks 1 --large_size 51 --small_size 5 --dims 128 --head_dropout 0.0 --dropout 0.1 --itr 1 --learning_rate 0.0005 --batch_size 128 --train_epochs 1 --patience 10 --des Exp --use_multi_scale False --small_kernel_merged False --apply_adv_aug False
