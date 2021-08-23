save_path=/home/v-weipeng/data1/Glat_xsum/save_models
input_dir=/home/v-weipeng/data1/Xsum/org_data
data_dir=/home/v-weipeng/data1/Xsum/processed_glat
tensor_file=/home/v-weipeng/data1/Glat_xsum/save_tensorboard
python train.py ${data_dir} --arch glat --noise full_mask --share-all-embeddings \
    --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir glat_plugins --batch-size 32 \
    --batch-size-valid 8 --tensorboard-logdir ${tensor_file} --skip-invalid-size-inputs-valid-test
