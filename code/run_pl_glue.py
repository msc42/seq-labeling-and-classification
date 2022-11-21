# based on huggingface/transformers
# commit:4b3ee9cbc53c6cf6cee6bfae86cc2c6ec0778ee5
# examples/text-classification/run_pl_glue.py

import argparse
import glob
import logging
import os
import time
from argparse import Namespace

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes

from custom_processor import CustomProcessor

logger = logging.getLogger(__name__)


class GLUETransformer(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        self.custom_processor = CustomProcessor()
        num_labels = len(self.custom_processor.get_labels())

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        self.labels = self.custom_processor.get_labels()

        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                if mode == 'train':
                    examples = self.custom_processor.get_train_examples(args.data_dir)
                elif mode == 'dev':
                    examples = self.custom_processor.get_dev_examples(args.data_dir)
                elif mode == 'test':
                    examples = self.custom_processor.get_test_examples(args.data_dir)

                features = convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.labels,
                    output_mode=args.glue_output_mode,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool=False) -> DataLoader:
        "Load datasets. Called after prepare data."

        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.hparams.glue_output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams.glue_output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(self.hparams.task, preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds, out_label_ids

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        f1, _, _ = self.calculate_f1_precision_recall(preds, targets)
        logs = ret["log"]
        return {"f1": f1, "val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        f1, _, _ = self.calculate_f1_precision_recall(predictions, targets)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"f1": f1, "avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def calculate_f1_precision_recall(self, predictions, targets):
        with open(os.path.join(self.hparams.output_dir, "tmp_classification_hyps.txt"), "w") as predictions_file:
            predictions_file.write("\n".join(str(prediction) for prediction in predictions))

        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = 2 * precision * recall / (precision + recall)
        print("precision, recall, f1:", precision, recall, f1)
        return f1, precision, recall

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--task",
            default="",
            type=str,
            required=True,
            help="The GLUE task to run",
        )
        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument(
            "--model_for_prediction",
            default='',
            type=str
        )

        return parser


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = GLUETransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = GLUETransformer(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.output_dir, '{epoch},{f1}'),
        monitor='f1',
        mode="max",
        save_top_k=1,
        period=0,
    )

    trainer = generic_train(model, args, checkpoint_callback=checkpoint_callback)

    # Optionally, predict on test set and write to output_dir
    if args.do_predict:
        if args.model_for_prediction:
            checkpoints = [args.model_for_prediction]
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "epoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.hparams.data_dir = args.data_dir
        model.hparams.output_dir = args.output_dir
        model.hparams.overwrite_cache = args.overwrite_cache
        return trainer.test(model)


if __name__ == "__main__":
    main()
