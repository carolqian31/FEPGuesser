# Overall
FEPGuesser is divided into three parts: PSVG, PassExBert, PassExBertVAE

dependencies is in requirements.txt

Please use the following code to download dependencies

```
pip install -r requirements.txt
```

# PSVG
The code in this folder is suitable for the generation of mixed vocabularies.

```
cd PSVG
python segmntr.py -s your_training_data_path.txt -t your_training_data_path.txt -o your_output_seg_save_folder_path -c gen_seg
```

in this way, you can get a folder of new password segments in *your_output_seg_save_folder_path*

Use the following code to obtain a wordlist txt file containing all words with a frequency higher than the threshold you set.

```
python get_wordlist_from_segment.py --folder_path your_output_seg_save_folder_path --threshold your_set_threshold --save_path wordlist_save_txt.txt
```

# PassExBert

