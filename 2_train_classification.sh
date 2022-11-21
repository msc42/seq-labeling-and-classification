#!/bin/sh

BERT_MODEL=bert-large-cased
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=1
NUM_RUNS=1
DATA_DIR=data/seq_classification
OUTPUT_DIR=experiments/classification
mkdir -p $OUTPUT_DIR

TASK=sst-2

train() {
mkdir -p $OUTPUT_DIR
python3 code/run_pl_glue.py \
	--data_dir $DATA_DIR \
	--task $TASK \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--train_batch_size $BATCH_SIZE \
	--gpus 1 \
	--do_train \
	--learning_rate 2e-5 \
	--do_predict
}

rm $DATA_DIR/cached*

for i in $(seq $NUM_RUNS); do
	train
done
