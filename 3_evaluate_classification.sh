#!/bin/sh

MODEL_FOR_PREDICTION=experiments/classification.ckpt

if [ -z "$MODEL_FOR_PREDICTION" ]; then
	echo "please set MODEL_FOR_PREDICTION variable"
	exit
fi

BERT_MODEL=bert-large-cased
MAX_LENGTH=128
BATCH_SIZE=32
TASK=sst-2
DATA_DIR=data/seq_classification

OUTPUT_DIR=experiments/tmp
mkdir -p $OUTPUT_DIR

predict() {
python3 code/run_pl_glue.py \
	--data_dir $DATA_DIR \
	--task $TASK \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--max_seq_length $MAX_LENGTH \
	--num_train_epochs 0 \
	--train_batch_size $BATCH_SIZE \
	--gpus 1 \
	--do_predict \
	--model_for_prediction $MODEL_FOR_PREDICTION \
	--overwrite_cache
}

mv $DATA_DIR/test.tsv $DATA_DIR/test_backup.tsv
cp $DATA_DIR/valid.tsv $DATA_DIR/test.tsv
predict
python3 code/error_detection_evaluator.py --mode export_correct_corrections --output_file $DATA_DIR/valid.correct $DATA_DIR/valid.tsv $OUTPUT_DIR/tmp_classification_hyps.txt

mv $DATA_DIR/test_backup.tsv $DATA_DIR/test.tsv
predict
python3 code/error_detection_evaluator.py --mode export_correct_corrections --output_file $DATA_DIR/test.correct $DATA_DIR/test.tsv $OUTPUT_DIR/tmp_classification_hyps.txt
