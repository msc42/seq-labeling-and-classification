#!/bin/sh

BERT_MODEL=bert-large-cased
MAX_LENGTH=128
BATCH_SIZE=32
LABEL_PATH=data/seq_labeling/labels.txt

case $1 in
	'detection_correction_with_classification_and_e2e_seq_labeling')
		WRONG_VALID=0
		WRONG_TEST=220
		MODEL_FOR_PREDICTION="experiments/e2e_seq_labeling.ckpt"
		DATA_DIR=data/seq_labeling_after_classification
	;;

	'detection_correction_with_classification_and_seq_labeling')
		WRONG_VALID=0
		WRONG_TEST=220
		MODEL_FOR_PREDICTION="experiments/seq_labeling.ckpt"
		DATA_DIR=data/seq_labeling_after_classification
	;;

	'detection_correction_with_e2e_seq_labeling')
		WRONG_VALID=0
		WRONG_TEST=0
		MODEL_FOR_PREDICTION="experiments/e2e_seq_labeling.ckpt"
		DATA_DIR=data/seq_labeling_e2e
	;;

	'correction_with_e2e_seq_labeling')
		WRONG_VALID=0
		WRONG_TEST=0
		MODEL_FOR_PREDICTION="experiments/e2e_seq_labeling.ckpt"
		DATA_DIR=data/seq_labeling
	;;

	'correction_with_seq_labeling')
		WRONG_VALID=0
		WRONG_TEST=0
		MODEL_FOR_PREDICTION="experiments/seq_labeling.ckpt"
		DATA_DIR=data/seq_labeling
	;;

	*)
		echo 'Please specify mode'
		exit
	;;
esac

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
python3 code/error_correction_evaluator.py  --add_number_wrong_before $WRONG_VALID $DATA_DIR/valid.txt $OUTPUT_DIR/tmp_labeling_hyps.txt
echo "Is WRONG_VALID $WRONG_VALID correct?"

mv $DATA_DIR/test_backup.txt $DATA_DIR/test.txt
predict
python3 code/error_correction_evaluator.py --add_number_wrong_before $WRONG_TEST $DATA_DIR/test.txt $OUTPUT_DIR/tmp_labeling_hyps.txt
echo "Is WRONG_TEST $WRONG_TEST correct?"
