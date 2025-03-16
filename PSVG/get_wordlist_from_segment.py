import os
import argparse

def get_wordlist_freq_from_file_list(file_list):
    """
    Get wordlist and frequency from txt file
    :param file_path: the path of the txt file
    :return: a list of tuple (word, frequency)
    """
    word_dict = {}
    for file_path in file_list:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                try:
                    word, freq = line.split('\t')
                except Exception as e:
                    print(e)
                    print(word)
                    continue
                word_dict[word] = word_dict.get(word, 0) + int(freq)
    return word_dict


def get_file_list_from_folder(folder):
    file_list = []
    for file in os.listdir(folder):
        if 'segment' in file:
            continue
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list


def setup():
    parser = argparse.ArgumentParser(description='get_wordlist_from_segment')
    parser.add_argument('--folder_path', type=str,  help='the segment folder path')
    parser.add_argument('--threshold', type=int, help='the threshold of add password patterns')
    parser.add_argument('--save_path', type=str, help='wordlist segment save path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup()

    folder_path = args.folder_path
    threshold = args.threshold
    file_list = get_file_list_from_folder(folder_path)
    word_dict = get_wordlist_freq_from_file_list(file_list)

    new_word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    words_chosen = []

    for word, freq in new_word_dict:
        if freq < threshold:
            break
        words_chosen.append(word)

    print(len(words_chosen))
    #
    # saved_path = '../PSVG/Rules/000webhost_segment/wordlist_segment_{}.txt'.format(threshold)
    save_path = args.save_path

    with open(saved_path, 'w') as f:
        for word in words_chosen:
            f.write(word + '\n')
