from typing import Callable, List, Union

from nltk import sent_tokenize, wordpunct_tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def calculate_sentence_length(sentence: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> int:
    return len(tokenizer.tokenize(sentence, add_special_tokens=True))


def split_sentence(long_sentence: str, max_seq_len: int, llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   word_tokenizer: Callable[[str], List[str]] = wordpunct_tokenize) -> List[str]:
    if calculate_sentence_length(long_sentence, llm_tokenizer) <= max_seq_len:
        return [long_sentence]
    word_bounds = []
    start_pos = 0
    for cur_word in word_tokenizer(long_sentence):
        found_idx = long_sentence[start_pos:].find(cur_word)
        if found_idx < 0:
            raise ValueError(f'The token "{cur_word}" is not found in the text "{long_sentence}".')
        word_bounds.append((found_idx + start_pos, found_idx + start_pos + len(cur_word)))
        start_pos = found_idx + start_pos + len(cur_word)
    if len(word_bounds) < 2:
        return [long_sentence]
    middle_idx = (len(word_bounds) - 1) // 2
    first_sentence_start = word_bounds[0][0]
    first_sentence_end = word_bounds[middle_idx][1]
    second_sentence_start = word_bounds[middle_idx + 1][0]
    second_sentence_end = word_bounds[-1][1]
    sentences = split_sentence(long_sentence[first_sentence_start:first_sentence_end], max_seq_len,
                               llm_tokenizer, word_tokenizer)
    sentences += split_sentence(long_sentence[second_sentence_start:second_sentence_end], max_seq_len,
                                llm_tokenizer, word_tokenizer)
    return sentences


def split_text_into_sentences(source_text: str, newline_as_separator: bool = True,
                              sent_tokenizer: Callable[[str], List[str]] = sent_tokenize,
                              word_tokenizer: Callable[[str], List[str]] = wordpunct_tokenize, max_seq_len: int = 512,
                              llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None) -> List[str]:
    if newline_as_separator:
        paragraphs = list(map(
            lambda it3: ' '.join(it3.split()).strip(),
            filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.strip(), source_text.split('\n')
                )
            )
        ))
    else:
        prepared_text = ' '.join(source_text.split()).strip()
        if len(prepared_text) == 0:
            paragraphs = []
        else:
            paragraphs = [prepared_text]
    if len(paragraphs) == 0:
        return []
    sentences = []
    for cur_paragraph in paragraphs:
        for it in sent_tokenizer(cur_paragraph):
            new_sentence = it.strip()
            if len(new_sentence) > 0:
                sentences.append(new_sentence)
    if llm_tokenizer is None:
        return sentences
    sentences_ = []
    for cur_sentence in sentences:
        sentences_ += split_sentence(cur_sentence, max_seq_len, llm_tokenizer, word_tokenizer)
    return sentences_
