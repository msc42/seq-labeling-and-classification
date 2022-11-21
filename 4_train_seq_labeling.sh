#!/bin/sh

BERT_MODEL=bert-large-cased
MAX_LENGTH=128
BATCH_SIZE=24
NUM_EPOCHS=1

case $1 in
	'seq_labeling')
		DATA_DIR=data/seq_labeling
		OUTPUT_DIR=experiments/seq_labeling/
	;;

	'e2e_seq_labeling')
		DATA_DIR=data/seq_labeling_e2e
		OUTPUT_DIR=experiments/e2e_seq_labeling/
	;;

	*)
		echo 'Please specify mode'
		exit
	;;
esac

LABEL_PATH=data/seq_labeling/labels.txt

mkdir -p $OUTPUT_DIR

train() {
	mkdir -p $OUTPUT_DIR/$1
	python3 code/run_pl_ner.py \
	--data_dir $DATA_DIR \
	--labels $LABEL_PATH \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR/$1 \
	--max_seq_length $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--train_batch_size $BATCH_SIZE \
	--gpus 1 \
	--do_train \
	--learning_rate 2e-5 \
	--do_predict
}

rm $DATA_DIR/cached*

for i in $(seq 1); do
	train tl_$i
done
