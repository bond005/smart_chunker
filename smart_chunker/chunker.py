import math
from typing import Callable, List, Tuple
import warnings

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import trange, tqdm

from smart_chunker.sentenizer import split_text_into_sentences, calculate_sentence_length

try:
    nltk.download("punkt_tab")
except LookupError:
    print("Looks like it is impossible to download or find''punkt_tab' package. Try to download it manually by 'nltk.download('punkt_tab')'")


class SmartChunker:
    def __init__(self, reranker_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'cpu',
                 sentence_tokenizer: Callable[[str], List[str]] = nltk.sent_tokenize, newline_as_separator: bool = True,
                 word_tokenizer: Callable[[str], List[str]] = nltk.wordpunct_tokenize,
                 max_chunk_length: int = 256, minibatch_size: int = 8, verbose: bool = False):
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer
        self.reranker_name = reranker_name
        self.device = device
        self.minibatch_size = minibatch_size
        self.max_chunk_length = max_chunk_length
        self.newline_as_separator = newline_as_separator
        self.verbose = verbose
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
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        if self.verbose:
            print(f'Model is loaded from the "{self.reranker_name}". The device is {self.device}.')

    def _get_pair(self, sentence_lengths: List[int],
                  split_index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        middle_pos = split_index + 1
        start_pos = middle_pos - 1
        end_pos = middle_pos + 1
        left_length = 0
        right_length = 0
        while start_pos >= 0:
            left_length += sentence_lengths[start_pos]
            if left_length > min(self.max_chunk_length, self.model_.config.max_position_embeddings // 2):
                break
            start_pos -= 1
        start_pos += 1
        if start_pos >= middle_pos:
            start_pos = middle_pos - 1
        while end_pos <= len(sentence_lengths):
            right_length += sentence_lengths[end_pos - 1]
            if right_length > min(self.max_chunk_length, self.model_.config.max_position_embeddings // 2):
                break
            end_pos += 1
        end_pos -= 1
        if end_pos < (middle_pos + 1):
            end_pos = middle_pos + 1
        return (start_pos, middle_pos), (middle_pos, end_pos)

    def _calculate_similarity_func(self, pairs: List[List[str]]) -> List[float]:
        if len(pairs) < 1:
            return []
        n_batches = math.ceil(len(pairs) / self.minibatch_size)
        scores = []
        for batch_idx in (trange(n_batches) if self.verbose else range(n_batches)):
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
        if len(scores) != len(pairs):
            err_msg = (f'The number of text pairs do not correspond to the number of calculated pair scores! '
                       f'{len(pairs)} != {len(scores)}')
            raise RuntimeError(err_msg)
        return scores

    def _find_chunks(self, sentences: List[str], split_scores: List[float],
                     start_pos: int, end_pos: int) -> List[Tuple[int, int]]:
        if len(sentences) < 1:
            return []
        if start_pos < 0:
            err_msg = f'The `start_pos` is wrong! Expected non-negative integer, got {start_pos}.'
            raise ValueError(err_msg)
        if end_pos > len(sentences):
            err_msg = f'The `end_pos` is wrong! Expected {len(sentences)} or less, got {end_pos}'
            raise ValueError(err_msg)
        if start_pos >= end_pos:
            err_msg = f'The chunk boundaries ({start_pos}, {end_pos}) are wrong!'
            raise ValueError(err_msg)
        if (len(sentences) < 2) or (start_pos == (end_pos - 1)):
            if self.verbose:
                print(f'Sentences from {start_pos} to {end_pos} form a new chunk.')
            return [(start_pos, end_pos)]
        if len(split_scores) != (len(sentences) - 1):
            err_msg = (f'The sentences do not correspond to the split scores! '
                       f'{len(sentences)} != {len(split_scores) + 1}')
            raise ValueError(err_msg)
        chunk_length = calculate_sentence_length(' '.join(sentences[start_pos:end_pos]), self.tokenizer_)
        if self.verbose:
            print(f'Sentences from {start_pos} to {end_pos} have a length of {chunk_length} tokens.')
        if chunk_length <= self.max_chunk_length:
            if self.verbose:
                print(f'Sentences from {start_pos} to {end_pos} form a new chunk.')
            return [(start_pos, end_pos)]
        best_split_idx = start_pos
        best_score = split_scores[start_pos]
        for cur_split_idx in range(start_pos + 1, end_pos - 1):
            cur_score = split_scores[cur_split_idx]
            if cur_score < best_score:
                best_score = cur_score
                best_split_idx = cur_split_idx
        chunks = self._find_chunks(sentences, split_scores, start_pos, best_split_idx + 1)
        chunks += self._find_chunks(sentences, split_scores, best_split_idx + 1, end_pos)
        return chunks

    def split_into_chunks(self, source_text: str) -> List[str]:
        source_text_ = source_text.strip()
        if len(source_text_) == 0:
            return []
        if calculate_sentence_length(source_text_, self.tokenizer_) <= self.max_chunk_length:
            return [source_text_]
        sentences = split_text_into_sentences(
            source_text,
            self.newline_as_separator, self.sentence_tokenizer, self.word_tokenizer,
            (2 * self.max_chunk_length) // 3, self.tokenizer_
        )
        if self.verbose:
            print(f'There are {len(sentences)} sentences in the text.')
        if len(sentences) < 2:
            return sentences
        sentence_lengths: List[int] = []
        if self.verbose:
            print(f'All sentences tokenization with {self.reranker_name} tokenizer is started.')
        for cur_sent in (tqdm(sentences) if self.verbose else sentences):
            sentence_lengths.append(len(self.tokenizer_.tokenize(cur_sent, add_special_tokens=True)))
        if len(sentences) != len(sentence_lengths):
            err_msg = (f'The sentence list size does not correspond to the sentence length list size! '
                       f'{len(sentences)} != {len(sentence_lengths)}.')
            raise RuntimeError(err_msg)
        if self.verbose:
            print(f'All sentences tokenization with {self.reranker_name} tokenizer is finished.')
        pairs = []
        for idx in range(len(sentences) - 1):
            left_chunk_bounds, right_chunk_bounds = self._get_pair(sentence_lengths, idx)
            pairs.append([
                ' '.join(sentences[left_chunk_bounds[0]:left_chunk_bounds[1]]),
                ' '.join(sentences[right_chunk_bounds[0]:right_chunk_bounds[1]])
            ])
            del left_chunk_bounds, right_chunk_bounds
        del sentence_lengths
        if self.verbose:
            print(f'Chunk candidates scoring with {self.reranker_name} model is started.')
        scores = self._calculate_similarity_func(pairs)
        del pairs
        if self.verbose:
            print(f'Chunk candidates scoring with {self.reranker_name} model is finished.')
        chunk_bounds = self._find_chunks(sentences, scores, 0, len(sentences))
        del scores
        chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds):
            chunk_candidate = ' '.join(' '.join(sentences[chunk_start:chunk_end]).strip().split())
            if chunk_end < len(sentences):
                other_chunk_candidate = ' '.join(' '.join(sentences[chunk_start:(chunk_end + 1)]).strip().split())
                if calculate_sentence_length(other_chunk_candidate, self.tokenizer_) <= self.max_chunk_length:
                    chunks.append(other_chunk_candidate)
                else:
                    chunks.append(chunk_candidate)
            else:
                chunks.append(chunk_candidate)
        return chunks
