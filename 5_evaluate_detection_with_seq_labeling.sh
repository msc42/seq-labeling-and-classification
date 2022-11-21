#!/bin/sh

MODEL_FOR_PREDICTION="experiments/e2e_seq_labeling.ckpt"

BERT_MODEL=bert-large-cased
MAX_LENGTH=128
BATCH_SIZE=32
DATA_DIR=data/seq_labeling_e2e
CLASSIFICATION_DATA_DIR=data/seq_classification
LABEL_PATH=data/seq_labeling/labels.txt

OUTPUT_DIR=experiments/tmp
mkdir -p $OUTPUT_DIR

predict() {
	python3 code/run_pl_ner.py \
	--data_dir $DATA_DIR \
	--labels $LABEL_PATH \
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

mv $DATA_DIR/test.txt $DATA_DIR/test_backup.txt
cp $DATA_DIR/valid.txt $DATA_DIR/test.txt
predict
python3 code/error_correction_evaluator.py --mode classification $DATA_DIR/valid.txt $OUTPUT_DIR/tmp_labeling_hyps.txt > $CLASSIFICATION_DATA_DIR/valid.seq_labeling
python3 code/error_detection_evaluator.py --mode export_correct_corrections --output_file $CLASSIFICATION_DATA_DIR/valid.correct_seq_labeling $CLASSIFICATION_DATA_DIR/valid.tsv $CLASSIFICATION_DATA_DIR/valid.seq_labeling

mv $DATA_DIR/test_backup.txt $DATA_DIR/test.txt
predict
python3 code/error_correction_evaluator.py  --mode classification $DATA_DIR/test.txt $OUTPUT_DIR/tmp_labeling_hyps.txt > $CLASSIFICATION_DATA_DIR/test.seq_labeling
python3 code/error_detection_evaluator.py --mode export_correct_corrections --output_file $CLASSIFICATION_DATA_DIR/test.correct_seq_labeling $CLASSIFICATION_DATA_DIR/test.tsv $CLASSIFICATION_DATA_DIR/test.seq_labeling
