# Original train
python -u run.py --task_name anomaly_detection --anomaly_ratio 1 --is_training 1 --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP  --seq_len 100 --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0001 --batch_size 128 --train_epochs 3  

# test
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode test --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP  --seq_len 100 --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  

# test second time
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode test_ori --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --seq_len 100  --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  

# Train Adv model
# # attack_target: Precision
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --adv_epoch 1 --mode train_adv_gen --attack_method Transformer_Adv --attack_target Precision  --anomaly_ratio 1 --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --seq_len 100  --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --epsilon 0.5


# Gen Test Adversarial Examples
# # # attack_target: Precision
# # # Random:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_test --attack_method Random --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP  --seq_len 100 --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1   --epsilon 0.05
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_test --attack_method FGSM --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP  --seq_len 100 --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  --epsilon 0.05    
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection  --anomaly_ratio 1 --mode gen_adv_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP  --seq_len 100 --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  --epsilon 0.05

# Test-Adversarial Examples
# # Precision
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode test_adv --attack_method FGSM --attack_target Precision  --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode test_adv --attack_method Transformer_Adv --attack_target Precision  --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  


# Visualization
# # Precision
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode visualize_adv --attack_method FGSM --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  
# # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --adv_epoch 3 --mode visualize_adv --attack_method Transformer_Adv --attack_target Precision  --anomaly_ratio 1 --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  


# Evaluation
# # Precision:
# # # FGSM
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode eval_adv --attack_method FGSM --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  
# # # PGD:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode eval_adv --attack_method PGD --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  
 # # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection  --anomaly_ratio 1 --mode eval_adv --attack_method Transformer_Adv --attack_target Precision  --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  


# Gen_Adv_Train
# # Precision
# # # FGSM:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode gen_adv_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  --epsilon 0.05 


# Defence_train 
# # Precision
# # # # Transformer_Adv
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --apply_adv_aug True

python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_train --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1 --apply_adv_aug False

# Defence_test
# # Precision
# # # Transformer_Adv:
python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  --apply_adv_aug True


python -u run_Adv.py --task_name anomaly_detection --anomaly_ratio 1 --mode defence_test --attack_method Transformer_Adv --attack_target Precision --root_path ./all_datasets/SMAP/ --model_id SMAP --model ModernTCN --data SMAP --enc_in 25 --c_out 25 --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 --large_size 13 --small_size 5 --dims 128   --learning_rate 0.0003 --batch_size 128 --train_epochs 1  --apply_adv_aug False
