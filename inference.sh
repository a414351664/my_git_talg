data_dir=/home/v-weipeng/data1/Xsum/processed_glat
checkpoint_path=/home/v-weipeng/data1/Glat_xsum/save_models/checkpoint_best.pt
OUTPUT_FILE=/home/v-weipeng/data1/Glat_xsum/outputs/output_best.txt
python inference.py ${data_dir} --path ${checkpoint_path} --user-dir glat_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang src --target-lang tgt \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset valid \
    --skip-invalid-size-inputs-valid-test > ${OUTPUT_FILE}

grep ^S $OUTPUT_FILE | cut -c 3- | sort -n | cut -f2- > /home/v-weipeng/data1/Glat_xsum/outputs/sort_source_best.txt
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- > /home/v-weipeng/data1/Glat_xsum/outputs/sort_hypo_best.txt
grep ^T $OUTPUT_FILE | cut -c 3- | sort -n | cut -f2- > /home/v-weipeng/data1/Glat_xsum/outputs/sort_tgt_best.txt
