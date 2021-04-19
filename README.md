# Transformer_Research
My personal Transformer architecture research

## khsBERT
BERT pretrained by 'Korean Hate Speech Detection' (https://www.kaggle.com/c/korean-hate-speech-detection).

- Need to refactoring
- Need to arguments re-arange
- Need to data-free code (Now only for Korean Hate Speech; KHS)
- Need to Optimizer & Learning rate scheduler setting

## P-Transformer
![Transformer_to_PTransformer](./PTransformer/img/Transformer_to_PTransformer.png)
Change Transformer network to Parallel mode. It inspired by Krashen's 'The Natural Order Hypothesis'.

- Need data sequence
- Need to data-free coda (Now only for WMT'16 de->en)
- Need beam search code to testing

## Reconstruct Feature
Use reconstruct BERT's last hidden feature map to input in other BERT.

- Need to refactoring
- Need to data-free code

## NER Conditional Augmentation
Expanding version of ConditionalBERT. Augment only person or organization information to augment and privacy protect.

- Need to refactoring
- Need to data-free code