input_dir=/home/v-weipeng/data1/Xsum/org_data
data_dir=/home/v-weipeng/data1/Xsum/processed_glat
#src=source_language
#tgt=target_language
python fairseq_cli/preprocess.py --source-lang src --target-lang tgt --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir}/ \
    --workers 32 --srcdict ${input_dir}/vocab.txt --tgtdict ${input_dir}/vocab.txt
