import math
from typing import List, Tuple
import warnings

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from smart_chunker.sentenizer import split_text_into_sentences, calculate_sentence_length


class SmartChunker:
    def __init__(self, language: str = 'ru', reranker_name: str = 'BAAI/bge-reranker-v2-m3',
                 newline_as_separator: bool = True,
                 device: str = 'cpu', max_chunk_length: int = 256, minibatch_size: int = 8, verbose: bool = False):
        self.language = language
        self.reranker_name = reranker_name
        self.device = device
        self.minibatch_size = minibatch_size
        self.max_chunk_length = max_chunk_length
        self.newline_as_separator = newline_as_separator
        self.verbose = verbose
        if self.language.strip().lower() not in {'ru', 'rus', 'russian', 'en', 'eng', 'english'}:
            raise ValueError(f'The language {self.language} is not supported!')
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.reranker_name, trust_remote_code=True)
        if self.device.lower().startswith('cuda'):
            try:
                self.model_ = AutoModelForSequenceClassification.from_pretrained(
                    self.reranker_name,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    attn_implementation='sdpa',
                    trust_remote_code=True
                )
            except BaseException as err:
                warnings.warn(str(err))
                self.model_ = AutoModelForSequenceClassification.from_pretrained(
                    self.reranker_name,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        else:
            self.model_ = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

    def _get_pair(self, sentences: List[str], split_index: int) -> List[str]:
        start_pos = 0
        middle_pos = split_index + 1
        end_pos = len(sentences)
        new_pair = [' '.join(sentences[start_pos:middle_pos]), ' '.join(sentences[middle_pos:end_pos])]
        left_length = calculate_sentence_length(new_pair[0], self.tokenizer_)
        right_length = calculate_sentence_length(new_pair[1], self.tokenizer_)
        while (left_length + right_length) >= self.model_.config.max_position_embeddings:
            if left_length > right_length:
                start_pos += 1
            else:
                end_pos -= 1
            if (start_pos >= middle_pos) or (end_pos <= middle_pos):
                start_pos = middle_pos - 1
                end_pos = middle_pos + 1
                del new_pair
                new_pair = [' '.join(sentences[start_pos:middle_pos]), ' '.join(sentences[middle_pos:end_pos])]
                break
            del new_pair
            new_pair = [' '.join(sentences[start_pos:middle_pos]), ' '.join(sentences[middle_pos:end_pos])]
            left_length = calculate_sentence_length(new_pair[0], self.tokenizer_)
            right_length = calculate_sentence_length(new_pair[1], self.tokenizer_)
        return new_pair

    def _calculate_similarity_func(self, sentences: List[str]) -> List[Tuple[int, float]]:
        if len(sentences) < 2:
            return []
        variants_of_split_index = list(filter(
            lambda idx: (calculate_sentence_length(' '.join(sentences[0:(idx + 1)]), self.tokenizer_) <=
                         self.max_chunk_length) and
                        (calculate_sentence_length(' '.join(sentences[(idx + 1):]), self.tokenizer_) <=
                         self.max_chunk_length) and
                        (calculate_sentence_length(' '.join(sentences[0:(idx + 1)]), self.tokenizer_) >=
                         self.max_chunk_length // 3) and
                        (calculate_sentence_length(' '.join(sentences[(idx + 1):]), self.tokenizer_) >=
                         self.max_chunk_length // 3),
            range(len(sentences) - 1)
        ))
        if len(variants_of_split_index) == 0:
            variants_of_split_index = list(filter(
                lambda idx: (calculate_sentence_length(' '.join(sentences[0:(idx + 1)]), self.tokenizer_) >=
                             self.max_chunk_length // 3) and
                            (calculate_sentence_length(' '.join(sentences[(idx + 1):]), self.tokenizer_) >=
                             self.max_chunk_length // 3),
                range(len(sentences) - 1)
            ))
            if len(variants_of_split_index) == 0:
                variants_of_split_index = list(range(len(sentences) - 1))
        pairs = [self._get_pair(sentences, idx) for idx in variants_of_split_index]
        n_batches = math.ceil(len(pairs) / self.minibatch_size)
        scores = []
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.minibatch_size
            batch_end = min(len(pairs), batch_start + self.minibatch_size)
            with torch.no_grad():
                inputs = self.tokenizer_(
                    pairs[batch_start:batch_end], return_tensors='pt',
                    padding=True, truncation=True, max_length=self.model_.config.max_position_embeddings
                )
                scores += self.model_(
                    **inputs.to(self.model_.device),
                    return_dict=True
                ).logits.float().cpu().numpy().flatten().tolist()
                del inputs
        return list(zip(variants_of_split_index, scores))

    def _find_chunks(self, sentences: List[str], start_pos: int, end_pos: int) -> List[str]:
        full_text_len = calculate_sentence_length(' '.join(sentences[start_pos:end_pos]), self.tokenizer_)
        if (full_text_len <= self.max_chunk_length) or ((end_pos - start_pos) < 2):
            if self.verbose:
                info_msg = f'Sentences from {start_pos} to {end_pos} form a new chunk.'
                print(info_msg)
            return [' '.join(sentences[start_pos:end_pos])]
        semantic_similarities = self._calculate_similarity_func(sentences[start_pos:end_pos])
        if len(semantic_similarities) == 0:
            if self.verbose:
                info_msg = f'Sentences from {start_pos} to {end_pos} form a new chunk.'
                print(info_msg)
            return [' '.join(sentences[start_pos:end_pos])]
        min_similarity_idx = semantic_similarities[0][0]
        min_similarity_val = semantic_similarities[0][1]
        for idx, val in semantic_similarities[1:]:
            if val < min_similarity_val:
                min_similarity_idx = idx
                min_similarity_val = val
        first_chunk = ' '.join(sentences[start_pos:(start_pos + min_similarity_idx + 1)])
        second_chunk = ' '.join(sentences[(start_pos + min_similarity_idx + 1):end_pos])
        all_chunks = []
        first_chunk_len = calculate_sentence_length(first_chunk, self.tokenizer_)
        second_chunk_len = calculate_sentence_length(second_chunk, self.tokenizer_)
        if self.verbose:
            info_msg = (f'Sentences from {start_pos} to {start_pos + min_similarity_idx + 1} '
                        f'have a length of {first_chunk_len} tokens.')
            print(info_msg)
            info_msg = (f'Sentences from {start_pos + min_similarity_idx + 1} to {end_pos} '
                        f'have a length of {second_chunk_len} tokens.')
            print(info_msg)
        if (min_similarity_idx == 0) or (first_chunk_len <= self.max_chunk_length):
            first_chunk_v2 = ' '.join(sentences[start_pos:(start_pos + min_similarity_idx + 2)])
            first_chunk_v2_len = calculate_sentence_length(first_chunk_v2, self.tokenizer_)
            if (first_chunk_v2_len <= self.max_chunk_length) and (first_chunk_v2 != first_chunk):
                all_chunks.append(first_chunk_v2)
                if self.verbose:
                    info_msg = f'Sentences from {start_pos} to {start_pos + min_similarity_idx + 2} form a new chunk.'
                    print(info_msg)
            else:
                all_chunks.append(first_chunk)
                if self.verbose:
                    info_msg = f'Sentences from {start_pos} to {start_pos + min_similarity_idx + 1} form a new chunk.'
                    print(info_msg)
        else:
            all_chunks += self._find_chunks(sentences, start_pos, start_pos + min_similarity_idx + 1)
        if ((start_pos + min_similarity_idx + 1) == (end_pos - 1)) or (second_chunk_len <= self.max_chunk_length):
            all_chunks.append(second_chunk)
            if self.verbose:
                info_msg = f'Sentences from {start_pos + min_similarity_idx + 1} to {end_pos} form a new chunk.'
                print(info_msg)
        else:
            all_chunks += self._find_chunks(sentences, start_pos + min_similarity_idx + 1, end_pos)
        return all_chunks

    def split_into_chunks(self, source_text: str) -> List[str]:
        source_text_ = source_text.strip()
        if len(source_text_) == 0:
            return []
        if calculate_sentence_length(source_text_, self.tokenizer_) <= self.max_chunk_length:
            return [source_text_]
        sentences = split_text_into_sentences(source_text, self.newline_as_separator, self.language,
                                              (2 * self.max_chunk_length) // 3, self.tokenizer_)
        if self.verbose:
            print(f'There are {len(sentences)} sentences in the text.')
        return self._find_chunks(sentences, 0, len(sentences))
