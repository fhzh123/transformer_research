import re
import emoji

def encoding_text(list_x, tokenizer, max_len):

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(r'<[^>]+>')

    def clean(x):
        x = pattern.sub(' ', x)
        x = x.strip()
        return x

    encoded_text_list = list_x.map(lambda x: tokenizer.encode(
        clean(str(x)),
        max_length=max_len,
        truncation=True
    ))
    return encoded_text_list