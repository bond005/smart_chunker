from argparse import ArgumentParser
import codecs
import os
import sys
from typing import List

import nltk
import razdel

try:
    from smart_chunker.chunker import SmartChunker
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from smart_chunker.chunker import SmartChunker


def word_tokenize_multi(s: str) -> List[str]:
    return nltk.wordpunct_tokenize(s)


def word_tokenize_ru(s: str) -> List[str]:
    return [it.text for it in razdel.tokenize(s)]


def sent_tokenize_multi(s: str) -> List[str]:
    return nltk.sent_tokenize(s)


def sent_tokenize_ru(s: str) -> List[str]:
    return [it.text for it in razdel.sentenize(s)]



def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=False,
                        default='BAAI/bge-reranker-v2-m3', help='The cross-encoder model name.')
    parser.add_argument('-i', '--input', dest='input_text', type=str, required=False,
                        default=os.path.join('demo', 'data', 'test_input.txt'),
                        help='The input file with source text.')
    parser.add_argument('-o', '--output', dest='output_text', type=str, required=True,
                        help='The output file with chunked text (each chunk - on a separate line).')
    parser.add_argument('--lang', dest='language', type=str, required=False, default='russian',
                        help='The language for text tokenization (Russian or English).')
    parser.add_argument('--device', dest='device', type=str, required=False, default='cuda:0',
                        help='The device for the cross-encoder inference.')
    parser.add_argument('--chunk', dest='chunk_length', type=int, required=False, default=200,
                        help='The maximal chunk length, i.e. maximal number of tokens per chunk.')
    parser.add_argument('--minibatch', dest='minibatch_size', type=int, required=False, default=32,
                        help='The mini-batch size.')
    parser.add_argument('--newline', dest='newline_as_separator', required=False, action='store_true',
                        default=False, help='Is a new line used to separate sentences?')
    parser.add_argument('--verbose', dest='verbose', required=False, action='store_true',
                        default=False, help='Will the smart chunker explain the chunking process?')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_text)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_text)
    if not os.path.isfile(output_fname):
        base_dir = os.path.dirname(output_fname)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                raise IOError(f'The directory "{base_dir}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        source_text = fp.read().strip()
    if len(source_text) == 0:
        raise IOError(f'The file "{input_fname}" is empty!')

    if args.language.strip().lower() in {'russian', 'ru', 'rus', 'russ'}:
        chunker = SmartChunker(
            reranker_name=args.model_name,
            newline_as_separator=args.newline_as_separator,
            device=args.device,
            max_chunk_length=args.chunk_length,
            minibatch_size=args.minibatch_size,
            verbose=args.verbose,
            sentence_tokenizer=sent_tokenize_ru,
            word_tokenizer=word_tokenize_ru
        )
    else:
        chunker = SmartChunker(
            reranker_name=args.model_name,
            newline_as_separator=args.newline_as_separator,
            device=args.device,
            max_chunk_length=args.chunk_length,
            minibatch_size=args.minibatch_size,
            verbose=args.verbose,
            sentence_tokenizer=sent_tokenize_multi,
            word_tokenizer=word_tokenize_multi
        )
    print('The smart chunker is prepared.')

    chunks = chunker.split_into_chunks(source_text)
    print('All chunks are successfully selected.')

    with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
        for cur_chunk in chunks:
            fp.write(cur_chunk + '\n\n')


if __name__ == '__main__':
    main()
