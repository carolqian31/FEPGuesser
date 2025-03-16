# Overall
FEPGuesser is divided into three parts: PSVG, PassExBert, PassExBertVAE

dependencies is in requirements.txt

Please use the following code to download dependencies

```
pip install -r requirements.txt
```

# PSVG
The code in this folder is suitable for the generation of mixed vocabularies.

your_training_data_path.txt is a file containing only passwords (one line one textual password).

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

go back to original folder and cd into PassExBert to train PassExBert.

```
cd ..
cd PassExBert
```
build mixed vocabulary
```
python build_pass_vocab_from_word_list.py --word_list_path wordlist_save_txt.txt --origin_vocab_path bert_base_cased/vocab.txt --output_vocab_path your_output_vocab_txt.txt --train_data_path your_training_data_path.txt
```

Now you get the mixed vocabulary in *your_output_vocab_txt.txt*

Specifically, we use bert natural language vocabulary. We upload it in *passExBert/bert_base_cased/vocab.txt*. 
You still need to download [pretrained model (pytorch_model.bin) in huggingface link](https://huggingface.co/google-bert/bert-base-cased/tree/main)

And you can also change it with your pretrained model and vocabulary.

Before you start to train PassExBert, you need to create model config, we show an example in *config_and_vocab/csdn/wordlist_segment_40.json*

Then, you start to train PassExBert, we give a training script example here.

```
python pretraining_pass.py -e 5 -b 128 -dv 0 1 -lr 1e-04 -str exBERT -sp ./storage/csdn/segment_40_e_5_b_256_mask_0.4 
          -config ./bert_base_cased/config.json config_and_vocab/csdn/wordlist_segment_40.json 
          -vocab config_and_vocab/csdn/wordlist_segment_40.txt -pm_p ./bert_base_cased/pytorch_model.bin 
          -dp train.txt 
          -ls 34 -p 1 -vp 0.1 -rd 10 -mr 0.4 -rir 0.1 -kr 0.05 -t_with_nlp_word
```

Finally, your trained PassExBert is in folder *./storage/csdn/segment_40_e_5_b_256_mask_0.4*

# PassExBertVAE

## Training 
Now, we can start to train PassExBertVAE.

Here is a training script.

```
python train_chunk.py --epoch 10 --batch_size 128 --dropout 0.3 --data csdn --magic_num 79 --experiment_info 
          passexbert_mask_0.4_first_and_last_layer_magic_79_epoch_10 
          --bert_dict_path your_trained_PassExBert_dict_path 
          --config ./bert_config_and_vocab/bert_base_cased/config.json bert_config_and_vocab/csdn/csdn/wordlist_segment_40.json 
          --passExBert_embed_type first_and_last_layer --device 0 1 
          --vocab mixed_vocab_path.txt 
          --max_seq_len 34
```

the trained PassExBertVAE checkpoints will be saved into folder *data/csdn_bert/passexbert_mask_0.4_first_and_last_layer_magic_79_epoch_10/*

## Sample

Use the following script to sample:

```
python PassExBert_sample.py --data csdn --vertex_num 2 --batch_size 512 --strategy gaussian 
          --origin_points_strategy dynamic_beam_random --num_sample 20000001 
          --vae_path segment_40_e_5_mask_0.4_first_and_last_layer_magic_79_epoch_10/best_dict_epoch9.pt 
          --sample_batch_size 4096 --config ./bert_config_and_vocab/bert_base_cased/config.json bert_config_and_vocab/csdn/wordlist_segment_40.json 
          --passExBert_embed_type first_and_last_layer --device 0 1 --vocab bert_config_and_vocab/csdn/wordlist_segment_40.txt 
          --bert_dict_path your_trained_PassExBert_path --sigma 0.01 --num_workers 16 
          --beam_size 5 --temperature 1.0 --step_size 0.03
```


Use this code to conduct cross-site attack experiments

```
python PassExBert_sample.py --data 000webhost --vertex_num 2 --batch_size 512 --strategy gaussian 
          --origin_points_strategy dynamic_beam_random --num_sample 10000000 
          --vae_path your_trained_vae_checkpoint_path
          --sample_batch_size 10240 --config ./bert_config_and_vocab/bert_base_cased/config.json bert_config_and_vocab/000webhost/wordlist_segment_200.json 
          --passExBert_embed_type first_and_last_layer --device 1 0 --vocab bert_config_and_vocab/000webhost/wordlist_segment_200.txt 
          --bert_dict_path your_trained_PassExBert_path --sigma 0.01 --num_workers 16 
          --beam_size 5 --temperature 1.0 --step_size 0.03 --test_data_path your_attacking_dataset_path
```