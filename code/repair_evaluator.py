#!/usr/bin/env python3
from __future__ import annotations

import argparse

'Assumption maximum two entities are corrected'


def generate_predicted_correction_target(tokens_and_labels,
                                         substitute_tokens_1, substitute_tokens_2,
                                         substitute_numbers_1, substitute_numbers_2, adapt_sep=True):
    predicted_correction_target: list[str] = []
    numbers: list[int] = []

    r1_location: list[int] = []
    r2_location: list[int] = []

    replaced_r1 = False
    replaced_r2 = False

    in_2_part = False

    for i, (token, label) in enumerate(tokens_and_labels, 1):
        if label == 'C':
            predicted_correction_target.append(token)
            numbers.append(i)

        elif label == 'R1':
            r1_location.append(i)
            if not replaced_r1:
                if substitute_tokens_1:
                    predicted_correction_target += substitute_tokens_1
                    numbers += [x for x in substitute_numbers_1]
                    replaced_r1 = True
                else:
                    predicted_correction_target.append(token)
                    numbers.append(i)
        elif label == 'R2':
            r2_location.append(i)
            if not replaced_r2:
                if substitute_tokens_2:
                    predicted_correction_target += substitute_tokens_2
                    numbers += [x for x in substitute_numbers_2]
                    replaced_r2 = True
                else:
                    predicted_correction_target.append(token)
                    numbers.append(i)
        elif adapt_sep and not in_2_part:
            in_2_part = True
            numbers.append(i)

    if adapt_sep and not in_2_part:
        numbers.append(i + 1)

    semicolon_position = i + 1
    arrow_position = i + 2

    if substitute_numbers_1:
        numbers += r1_location
        numbers.append(arrow_position)
        numbers += [x for x in substitute_numbers_1]

    if substitute_numbers_1 and substitute_numbers_2:
        numbers.append(semicolon_position)

    if substitute_numbers_2:
        numbers += r2_location
        numbers.append(arrow_position)
        numbers += [x for x in substitute_numbers_2]

    return predicted_correction_target, numbers


def generate_predicted_extraction_target(replaced_tokens_1, replaced_tokens_2,
                                         substitute_tokens_1, substitute_tokens_2):
    substitute_tokens_1_str = ' '.join(substitute_tokens_1)
    substitute_tokens_2_str = ' '.join(substitute_tokens_2)
    replaced_tokens_1_str = ' '.join(replaced_tokens_1)
    replaced_tokens_2_str = ' '.join(replaced_tokens_2)

    if substitute_tokens_1 and substitute_tokens_2:
        return (replaced_tokens_1_str + ' -> ' + substitute_tokens_1_str,
                replaced_tokens_2_str + ' -> ' + substitute_tokens_2_str)
    if substitute_tokens_1:
        return (replaced_tokens_1_str + ' -> ' + substitute_tokens_1_str, '')
    if substitute_tokens_2:
        return (replaced_tokens_2_str + ' -> ' + substitute_tokens_2_str, '')

    return ('', '')


def generate_predicted_targets(tokens_and_labels):
    replaced_tokens_1: list[str] = []
    replaced_tokens_2: list[str] = []
    substitute_tokens_1: list[str] = []
    substitute_tokens_2: list[str] = []
    substitute_numbers_1: list[int] = []
    substitute_numbers_2: list[int] = []
    for i, (token, label) in enumerate(tokens_and_labels, 1):
        if label == 'R1':
            replaced_tokens_1.append(token)
        elif label == 'R2':
            replaced_tokens_2.append(token)
        elif label == 'S1':
            substitute_tokens_1.append(token)
            substitute_numbers_1.append(i)
        elif label == 'S2':
            substitute_tokens_2.append(token)
            substitute_numbers_2.append(i)

    predicted_correction_target, numbers = generate_predicted_correction_target(tokens_and_labels,
                                                                                substitute_tokens_1,
                                                                                substitute_tokens_2,
                                                                                substitute_numbers_1,
                                                                                substitute_numbers_2)

    predicted_extraction_target = generate_predicted_extraction_target(replaced_tokens_1, replaced_tokens_2,
                                                                       substitute_tokens_1, substitute_tokens_2)
    numbers_target = ' '.join(str(number) for number in numbers)
    labels_target = ' '.join(label for _, label in tokens_and_labels)

    return ' '.join(predicted_correction_target), predicted_extraction_target, numbers_target, labels_target


def calculate_correction_acc(tokens_labels_file_path, predictions):
    tokens_labels_all = []
    with open(tokens_labels_file_path, 'r') as tokens_labels_file:
        tokens_labels = []
        for line in tokens_labels_file:
            line = line.strip()
            if line:
                token, label = line.split(' ')
                tokens_labels.append((token, label))
            else:
                tokens_labels_all.append(tokens_labels)
                tokens_labels = []

    if len(tokens_labels_all) != len(predictions):
        print('warning: prediction length unequal file length, okay in sanity runs')

    errors = 0
    for tokens_labels, prediction in zip(tokens_labels_all, predictions):
        tokens_and_labels_prediction = [(token, label) for (token, _), label in zip(tokens_labels, prediction)]

        corrected_hyp, extracted_hyp, _, _ = generate_predicted_targets(tokens_and_labels_prediction)
        corrected_ref, extracted_ref, _, _ = generate_predicted_targets(tokens_labels)

        error_in_extraction_order_1 = extracted_hyp[0] != extracted_ref[0] or extracted_hyp[1] != extracted_ref[1]
        error_in_extraction_order_2 = extracted_hyp[0] != extracted_ref[1] or extracted_hyp[1] != extracted_ref[0]

        if corrected_hyp != corrected_ref or (error_in_extraction_order_1 and error_in_extraction_order_2):
            errors += 1

    return 1 - (errors / len(predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source_file', type=str)
    parser.add_argument('predictions_file', type=str)

    args = parser.parse_args()

    predictions: list[list[str]] = []
    prediction: list[str] = []
    with open(args.predictions_file, 'r') as predictions_file:
        for line in predictions_file:
            line = line.strip()
            if line:
                prediction.append(line)
            else:
                predictions.append(prediction)
                prediction = []

    calculate_correction_acc(args.source_file, predictions)
