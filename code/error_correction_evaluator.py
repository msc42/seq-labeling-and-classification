#!/usr/bin/env python3

from __future__ import annotations

import argparse
from enum import Enum
import re

from repair_evaluator import generate_predicted_targets


class Mode(str, Enum):
    STAT = 'stat',
    CORRECT = 'correct'
    ERROR = 'error'
    CLASSIFICATION = 'classification'

    def __str__(self):
        return self.value


def calculate_results(ref_file_path, hyp_file_path, mode, add_number_wrong_before, case_sensitiv=False):
    total = add_number_wrong_before
    errors = add_number_wrong_before
    errors_correction = add_number_wrong_before
    errors_extraction = add_number_wrong_before

    with open(ref_file_path, 'r') as ref_file, \
            open(hyp_file_path, 'r') as hyp_file:
        tokens_and_labels_ref = []
        tokens_and_labels_hyp = []

        for ref_line, hyp_line in zip(ref_file, hyp_file):
            ref_line = ref_line.strip()
            hyp_line = hyp_line.strip()

            if ref_line and not hyp_line or not hyp_line and ref_line:
                print(ref_line)
                print(ref_line)
                raise ValueError()

            if ref_line:
                token, label_ref = ref_line.split()
                label_hyp = hyp_line

                tokens_and_labels_ref.append((token, label_ref))
                tokens_and_labels_hyp.append((token, label_hyp))
                continue

            total += 1

            if mode == Mode.CLASSIFICATION:
                no_correction_detected = all(labels == 'C' for _, labels in tokens_and_labels_hyp)
                print(1 if no_correction_detected else 0)

            corrected_seq_ref, extracted_tuple_ref, _, _ = generate_predicted_targets(tokens_and_labels_ref)
            corrected_seq_hyp, extracted_tuple_hyp, _, _ = generate_predicted_targets(tokens_and_labels_hyp)

            tokens_and_labels_ref = []
            tokens_and_labels_hyp = []

            error = False

            if not case_sensitiv:
                corrected_seq_ref = corrected_seq_ref.lower()
                corrected_seq_hyp = corrected_seq_hyp.lower()

            if corrected_seq_ref != corrected_seq_hyp:
                errors_correction += 1
                error = True

            ref_extraction_1 = extracted_tuple_ref[0]
            ref_extraction_2 = extracted_tuple_ref[1]
            hyp_extraction_1 = extracted_tuple_hyp[0]
            hyp_extraction_2 = extracted_tuple_hyp[1]

            if not case_sensitiv:
                ref_extraction_1 = ref_extraction_1.lower()
                ref_extraction_2 = ref_extraction_2.lower()
                hyp_extraction_1 = hyp_extraction_1.lower()
                hyp_extraction_2 = hyp_extraction_2.lower()

            if re.fullmatch('(.*) -> \\1', hyp_extraction_1):
                hyp_extraction_1 = ''

            if re.fullmatch('(.*) -> \\1', hyp_extraction_2):
                hyp_extraction_2 = ''

            error_in_extraction_order_1 = ref_extraction_1 != hyp_extraction_1 or ref_extraction_2 != hyp_extraction_2
            error_in_extraction_order_2 = ref_extraction_1 != hyp_extraction_2 or ref_extraction_2 != hyp_extraction_1

            if error_in_extraction_order_1 and error_in_extraction_order_2:
                errors_extraction += 1
                error = True

            if error:
                errors += 1

            if mode == Mode.ERROR and error:
                print(f'{hyp_line} vs {ref_line}')
            elif mode == Mode.CORRECT and not error:
                print(f'{hyp_line}')

    return 1 - (errors / total), 1 - (errors_correction / total), 1 - (errors_extraction / total)


def get_latex_str(results):
    return f'{results[1] * 100:.2f}\,\% & {results[2] * 100:.2f}\,\% & {results[0] * 100:.2f}\,\%'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_file', type=str)
    parser.add_argument('hyp_file', type=str)

    parser.add_argument('--mode', type=Mode, default=Mode.STAT, choices=tuple(Mode))
    parser.add_argument('--add_number_wrong_before', type=int, default=0, help='for pipeline approach')

    args = parser.parse_args()

    results = calculate_results(args.ref_file, args.hyp_file, args.mode, args.add_number_wrong_before)

    if args.mode == Mode.STAT:
        print(results)
        print(get_latex_str(results))
