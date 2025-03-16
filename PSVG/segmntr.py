#!/usr/bin/env python3
"""
This is to get the segments of a password.
"""
import argparse
import os.path
import sys
import time
from collections import defaultdict
from typing import TextIO

from lib_trainer.detection_rules.alpha_detection import alpha_detection
from lib_trainer.detection_rules.digit_detection import digit_detection
from lib_trainer.detection_rules.keyboard_walk import detect_keyboard_walk
from lib_trainer.detection_rules.multiword_detector import MultiWordDetector
from lib_trainer.detection_rules.context_sensitive_detection import context_sensitive_detection
# from lib_trainer.future_research.my_context_detection import detect_context_sections
from lib_trainer.future_research.my_leet_detector import AsciiL33tDetector
from lib_trainer.future_research.my_multiword_detector import MyMultiWordDetector
from lib_trainer.detection_rules.other_detection import other_detection
from lib_trainer.detection_rules.year_detection import year_detection


def v41seg(training: TextIO, test_set: TextIO, save2: TextIO) -> None:
    if not save2.writable():
        raise Exception(f"{save2.name} is not writable")

    multiword_detector = MultiWordDetector()
    for password in training:
        password = password.strip("\r\n")
        multiword_detector.train(password)
    training.close()

    pwd_counter = defaultdict(int)
    for password in test_set:
        password = password.strip("\r\n")
        pwd_counter[password] += 1
    test_set.close()

    for password, num in pwd_counter.items():
        section_list, found_walks = detect_keyboard_walk(password)
        _ = year_detection(section_list)
        """
        Note that there is a bug in context_sensitive_detection
        I have fixed that and add a test case in unit_tests folder
        """
        _ = context_sensitive_detection(section_list)
        _, _ = alpha_detection(section_list, multiword_detector)
        _ = digit_detection(section_list)
        _ = other_detection(section_list)
        info = [password, f"{num}"]
        npass = ""
        for sec, tag in section_list:
            npass += sec
            info.append(sec)
            info.append(tag)
        if password.lower() != npass.lower():
            print(password)
            print(section_list)
            raise Exception("neq")
        print("\t".join(info), end="\n", file=save2)
    pass


def v41_seg_test(training, save2, testing=None):
    if not save2.writable():
        raise Exception(f"{save2.name} is not writable")

    multiword_detector = MultiWordDetector()
    for password in training:
        password = password.strip("\r\n")
        multiword_detector.train(password)

    if testing is not None:
        training.close()

        pwd_counter = defaultdict(int)
        for password in testing:
            password = password.strip("\r\n")
            pwd_counter[password] += 1
        testing.close()

    else:
        training.seek(0)
        pwd_counter = defaultdict(int)
        for password in training:
            password = password.strip("\r\n")
            pwd_counter[password] += 1
        training.close()

    count = 0
    for password, num in pwd_counter.items():
        section_list, found_list, detected_keyboards = detect_keyboard_walk(password)
        _ = year_detection(section_list)
        """
        Note that there is a bug in context_sensitive_detection
        I have fixed that and add a test case in unit_tests folder
        """
        _ = context_sensitive_detection(section_list)
        _, _ = alpha_detection(section_list, multiword_detector)
        _ = digit_detection(section_list)
        _ = other_detection(section_list)
        info = [password, f"{num}"]
        npass = ""
        for sec, tag in section_list:
            npass += sec
            info.append(sec)
            info.append(tag)
        if password.lower() != npass.lower():
            print(password)
            print(section_list)
            raise Exception("neq")
        print("\t".join(info), end="\n", file=save2)
        count += 1

        if count % 1000000 == 0:
            print("process {} passwords".format(count))
    pass

def generate_bpe_seg(training: TextIO, save_folder: str):
    multiword_detector = MyMultiWordDetector()
    for password in training:
        password = password.strip("\r\n")
        multiword_detector.train(password)

    multiword_detector.new_lendict()
    multiword_dict_order_by_len = multiword_detector.lendict

    l33t_detector = AsciiL33tDetector(multiword_detector)
    l33t_detector.init_l33t(training.name, "ascii")
    pwd_counter = defaultdict(int)

    training.seek(0)
    for password in training:
        password = password.strip("\r\n")
        pwd_counter[password] += 1

    training.close()

    print("rules done")
    year_dict = {}
    leets_dict = {}
    context_dict = {}
    alpha_dict = {}
    digit_dict = {}
    special_dict = {}
    keyboard_walk_dict = {}
    count = 0
    for password, num in pwd_counter.items():
        # section_list = [(password, None)]
        section_list, found_walks, _ = detect_keyboard_walk(password)
        for walk in found_walks:
            keyboard_walk_dict[walk] = keyboard_walk_dict.get(walk, 0) + 1
        years = year_detection(section_list)
        for year in years:
            year_dict[year] = year_dict.get(year, 0) + 1
        # section_list, _ = detect_context_sections(section_list)
        context_sensitive_list = context_sensitive_detection(section_list)
        for context_sensitive in context_sensitive_list:
            context_dict[context_sensitive] = context_dict.get(context_sensitive, 0) + 1

        section_list, leet_list, _ = l33t_detector.parse_sections(section_list)
        for leet in leet_list:
            leets_dict[leet] = leets_dict.get(leet, 0) + 1

        section_list, extracted_letters, extracted_mask, extracted_digits, extracted_specials = \
            multiword_detector.parse_sections(section_list)
        for alpha in extracted_letters:
            alpha_dict[alpha] = alpha_dict.get(alpha, 0) + 1
        for digit in extracted_digits:
            digit_dict[digit] = digit_dict.get(digit, 0) + 1
        for special in extracted_specials:
            special_dict[special] = special_dict.get(special, 0) + 1

        count += 1
        if count % 100000 == 0:
            print(count)
        # info = [password, f"{num}"]
        # npass = ""
        # for sec, tag in section_list:
        #     npass += sec
        #     info.append(sec)
        #     info.append(tag)
        # if password.lower() != npass.lower():
        #     # Note that we'll not lower X
        #     # therefore, the best way is to compare password.lower
        #     # with npass.lower
        #     print(password)
        #     print(section_list)
        #     raise Exception("neq")
        # print("\t".join(info), end="\n")

    sort_and_save_dict(year_dict, save_folder, "year")
    sort_and_save_dict(leets_dict, save_folder, "leets")
    sort_and_save_dict(context_dict, save_folder, "context")
    sort_and_save_dict(alpha_dict, save_folder, "alpha")
    sort_and_save_dict(digit_dict, save_folder, "digit")
    sort_and_save_dict(special_dict, save_folder, "special")
    sort_and_save_dict(keyboard_walk_dict, save_folder, "keyboard")

    print("done")


def sort_and_save_dict(dict, save_folder, dict_name):
    new_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print("{}\t{} lines".format(dict_name, len(new_dict)))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

        print("create folder {}".format(save_folder))
    save_path = os.path.join(save_folder, dict_name + ".txt")
    with open(save_path, 'w') as f:
        for key, value in new_dict:
            f.write("{}\t{}\n".format(key, value))
    return new_dict

def l33tseg(training: TextIO, test_set: TextIO, save2: TextIO) -> None:
    if not save2.writable():
        raise Exception(f"{save2.name} is not writable")

    multiword_detector = MyMultiWordDetector()
    for password in training:
        password = password.strip("\r\n")
        multiword_detector.train(password)
    training.close()
    l33t_detector = AsciiL33tDetector(multiword_detector)
    l33t_detector.init_l33t(training.name, "ascii")
    pwd_counter = defaultdict(int)
    for password in test_set:
        password = password.strip("\r\n")
        pwd_counter[password] += 1
    test_set.close()
    for password, num in pwd_counter.items():
        section_list = [(password, None)]
        _ = year_detection(section_list)
        # section_list, _ = detect_context_sections(section_list)
        _ = context_sensitive_detection(section_list)
        section_list, _, _ = l33t_detector.parse_sections(section_list)
        section_list, _, _, _, _ = multiword_detector.parse_sections(section_list)
        info = [password, f"{num}"]
        npass = ""
        for sec, tag in section_list:
            npass += sec
            info.append(sec)
            info.append(tag)
        if password.lower() != npass.lower():
            # Note that we'll not lower X
            # therefore, the best way is to compare password.lower
            # with npass.lower
            print(password)
            print(section_list)
            raise Exception("neq")
        print("\t".join(info), end="\n", file=save2)
    pass


def main():
    v41, l33t, gen_seg, gen_pcfg_seg = ["v41", "l33t", "gen_seg", "gen_pcfg_seg"]
    cli = argparse.ArgumentParser("Find the structures of passwords in test set")
    cli.add_argument("-s", "--src", dest="training", required=True, type=argparse.FileType("r"),
                     help="training set")
    cli.add_argument("-t", "--tar", dest="testing", required=False, type=argparse.FileType("r"),
                     help="testing set")
    cli.add_argument("-o", "--output", dest="save2", required=False, type=str,
                     help="save output here")
    cli.add_argument("-c", "--choice", dest="choice", required=False, choices=[v41, l33t, gen_seg, gen_pcfg_seg], type=str, default="gen_seg",
                     help="use v41 or v41-with-l33t")
    args = cli.parse_args()
    choice, training, testing = \
        args.choice, args.training, args.testing # type: str, TextIO, TextIO
    save2 = args.save2
    if choice == v41:
        v41seg(training=training, test_set=testing, save2=save2)
    elif choice == l33t:
        l33tseg(training=training, test_set=testing, save2=save2)
    elif choice == gen_seg:
        generate_bpe_seg(training=training, save_folder=save2)
    elif choice == gen_pcfg_seg:
        with open(save2, "w") as f:
            v41_seg_test(training=training, save2=f, testing=args.testing)
    else:
        print("Unknown method or method has not been implemented", file=sys.stderr)
        sys.exit(-1)


if __name__ == '__main__':
    main()
