SAVE_DIR=~/vit_10b_fsdp_example_ckpts  # this can be any directory (it doesn't need to be a shared one across nodes)

python3 -u ~/vit_10b_fsdp_example/run_vit_training.py \
  --fake_data \
  --ckpt_dir ${SAVE_DIR} \
  --image_size 224 \
  --patch_size 14 \
  --embed_dim 5120 \
  --mlp_ratio 4.0 \
  --num_heads 32 \
  --num_blocks 32 \
  --batch_size 1024 \
  --num_epochs 300 \
  --lr 1e-3 \
  --weight_decay 0.1 \
  --clip_grad_norm 1.0 \
  --warmup_steps 10000 \
  --log_step_interval 20 \
  --shard_on_cpu \
  # 2>&1 | tee ${SAVE_DIR}/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
