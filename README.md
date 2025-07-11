[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-green.svg)

# Smart-Chunker
This **smart chunker** is a semantic chunker to prepare a  long document for retrieval augmented generation (RAG).

Unlike a usual chunker, it does not split the text into identical groups of N tokens. Instead, it uses a cross-encoder to calculate the similarity function between neighboring sentences and divides the text based on the most significant  boundaries of semantic transitions, i.e. minima in the above-mentioned similarity function.

The `BAAI/bge-reranker-v2-m3`, or any other model that supports the  AutoModelForSequenceClassification interface, should be used  as a cross encoder.

Key idea
--------

A key element of a RAG (Retrieval Augmented Generation) system is the module for searching the text corpus to find context relevant to the user's query for the LLM. The text corpus needs to be structured, i.e., specially prepared to make searching more efficient. One important stage of this structuring is "chunking" - breaking down large texts into smaller fragments or chunks. Often, this division is done rather primitively: the text is split into groups of tokens of equal size without considering semantics. Humans, however, improve text readability by dividing it differently: into paragraphs, which are semantically homogeneous text segments consisting of one or several sentences. The main idea of the smart chunker is to reproduce the semantic text division characteristic of human chunking, which improves the quality of text corpus structuring and positively affects the RAG system's performance.

The smart chunking algorithm consists of the following steps:

**Step 1.** The entire text is split into individual sentences. If the text is in English, the `sent_tokenize` function from [the nltk library](https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html) is used for sentence segmentation. If the text is in Russian, the `sentenize` function from [the razdel library](https://github.com/natasha/razdel) is used. Additionally, a newline character can serve as an optional criterion for ending a sentence.

**Step 2.** Multiple variants of dividing the text into two chunks are generated:

- *variant 1*: the first chunk includes the first sentence, and the second chunk includes the remaining sentences from the second to the last;
- *variant 2*: the first chunk includes the first and second sentences, and the second chunk includes the remaining sentences from the third to the last;
- *variant 3*: the first chunk includes the first, second, and third sentences, and the second chunk includes the remaining sentences from the fourth to the last;
- and so on...

**Step 3.** Using a cross-encoder, the semantic similarity between the first and second chunks is calculated for each text division variant.

**Step 4.** The final variant for dividing the text into two chunks is the one where the semantic similarity between the first and second chunks is minimal.

For each of the two identified chunks, this procedure is repeated recursively until the chunks are sufficiently small (less than a predefined maximum allowable chunk length). The resulting chunks may overlap by one sentence (although this is not always the case).  

Installing
----------


For installation, you need to Python 3.10 or later. You can install the **Smart-Chunker** from the [PyPi](https://pypi.org/project/smart-chunker) using the following command:

```
python -m pip install smart-chunker
```

If you want to install the **Smart-Chunker** in a Python virtual environment, you don't need to use `sudo`, but before installing, you will need to activate this virtual environment. For example, you can do this by using `conda activate your_favourite_environment` in the Linux terminal, or in the Anaconda Prompt for Microsoft Windows).

Also, 

To build this project from sources, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/smart_chunker.git
cd smart_chunker
python -m pip install .
```

In this case, you can run the unit tests to check workability and environment setting correctness:

```
python setup.py test
```

or

```
python -m unittest
```

For these tests, the `BAAI/bge-reranker-v2-m3` model will be used by default, and this model will be automatically downloaded from [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3). However, if your internet connection is unstable, you can download all the necessary files for this model and store them in the `tests/testdata/bge_reranker` subdirectory.

Usage
-----

After installing the **Smart-Chunker**, you can use it as a Python package in your projects. For example, you can create a new smart chunker for English using the [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and apply it to some English text as follows:

```python
from smart_chunker.chunker import SmartChunker

chunker = SmartChunker(
    language='en',
    reranker_name='BAAI/bge-reranker-v2-m3',
    newline_as_separator=False,
    device='cuda:0',
    max_chunk_length=250,
    minibatch_size=8,
    verbose=True
)
demo_text = 'There are many different approaches to Transformer fine-tuning. ' \
            'First, there is a development direction dedicated to the modification of ' \
            'the loss function and a specific problem statement. For example, training problem ' \
            'could be set as machine reading comprehence (question answering) instead of ' \
            'the standard sequence classification, or focal loss, dice loss and other things ' \
            'from other deep learning domains could be used instead of the standard ' \
            'cross-entropy loss function. Second, there are papers devoted to BERT extension, ' \
            'related to adding more input information from the knowledge graph, ' \
            'morpho-syntactic parsers and other things. Third, there is a group of algorithms ' \
            'associated with changing the learning procedure, such as metric learning ' \
            '(contrastive learning). Each direction has its own advantages and disadvantages, ' \
            'but the metric learning seems the most promising to us. Because the goal of ' \
            'any training is not to overfit the training sample and not just to take the top of ' \
            'the leaderboard on a particular test sample from the general population, ' \
            'but to ensure the highest generalization ability on the general population ' \
            'as a whole. High generalization ability is associated with good separation in ' \
            'the feature space. A good separation is possible when objects of ' \
            'different classes form sufficiently compact regions in our space. And methods of ' \
            'contrastive learning achieve better separation. Our goal is to test, on the basis of ' \
            'the RuNNE competition (Artemova et al., 2022), how true are these theoretical ' \
            'considerations in practice and how much will the use of comparative learning in ' \
            'BERT’s fine tuning allow us to build more compact high-level representations of ' \
            'different classes of named entities and, as a result, improve the quality of recognition ' \
            'of named entities.'
chunks = chunker.split_into_chunks(source_text=demo_text)
for cur_chunk in chunks: print(cur_chunk + '\n')
```

During the execution process, you may see the following message in the log:

```text
There are 11 sentences in the text.
Sentences from 0 to 6 have a length of 188 tokens.
Sentences from 6 to 11 have a length of 192 tokens.
Sentences from 0 to 7 form a new chunk.
Sentences from 6 to 11 form a new chunk.
```

As a result of the execution, you will see two chunks (with overlapping content in one sentence):

- **The first chunk:** *There are many different approaches to Transformer fine-tuning. First, there is a development direction dedicated to the modification of the loss function and a specific problem statement. For example, training problem could be set as machine reading comprehence (question answering) instead of the standard sequence classification, or focal loss, dice loss and other things from other deep learning domains could be used instead of the standard cross-entropy loss function. Second, there are papers devoted to BERT extension, related to adding more input information from the knowledge graph, morpho-syntactic parsers and other things. Third, there is a group of algorithms associated with changing the learning procedure, such as metric learning (contrastive learning). Each direction has its own advantages and disadvantages, but the metric learning seems the most promising to us. Because the goal of any training is not to overfit the training sample and not just to take the top of the leaderboard on a particular test sample from the general population, but to ensure the highest generalization ability on the general population as a whole.*

- **The second chunk:** *Because the goal of any training is not to overfit the training sample and not just to take the top of the leaderboard on a particular test sample from the general population, but to ensure the highest generalization ability on the general population as a whole. High generalization ability is associated with good separation in the feature space. A good separation is possible when objects of different classes form sufficiently compact regions in our space. And methods of contrastive learning achieve better separation. Our goal is to test, on the basis of the RuNNE competition (Artemova et al., 2022), how true are these theoretical considerations in practice and how much will the use of comparative learning in BERT’s fine tuning allow us to build more compact high-level representations of different classes of named entities and, as a result, improve the quality of recognition of named entities.*

Demo
----

In the `demo` subdirectory you can see the **text_to_chunks.py** script and the **data** subdirectory. The `text_to_chunks script.py` is a wrapper for the aforementioned `SmartChunker` class. You need to call this script as follows:

```shell
python text_to_chunks.py \
    -m /path/to/cross-encoder \
    -i /path/to/input/file/with/large/text.txt \
    -o /path/to/resulted/file/with/chunks.txt \
    --lang some_language \
    --device cuda:0 \
    --chunk 300 \
    --minibatch 16 \
    --newline \
    --verbose
```

The argument **-m** is a name of cross-encoder model (for example, `BAAI/bge-reranker-v2-m3` or `Alibaba-NLP/gte-multilingual-reranker-base`). This can be the name of the model in the HuggingFace Hub or the path to a folder containing the model on a user's local device.

The argument **-i** is an input file with text that needs to be split into chunks.

The argument **-o** is an output file that will contain the result of splitting the source text into chunks (each chunk is on a separate line).

The argument **--lang** is a language of the source text (`russian` or `english`).

The argument **--device** specifies a device to use for inference (I recommend setting `cuda` or `cuda:0`, but `cpu` can also be used).

The argument **--chunk** limits the maximum size of a chunk (in terms of tokens of the tokenizer used by the specified cross-encoder).

The argument **--minibatch** determines the size of the mini-batch when calculating the semantic similarity between neighboring "chunk candidates" with the cross-encoder. The larger the size, the faster the calculation, but it also requires more video memory.

The argument **--newline** indicates whether to use the newline character as a criterion for sentence termination when preliminary dividing text into sentences (before dividing it into chunks as sentence groups).

The argument **--verbose** specifies the need to log the steps of the text splitting algorithm into chunks (if this argument is not specified, then the algorithm works "silently").

Breaking Changes
-------

**Breaking changes in version 0.0.2**
- The boundaries of chunks can now overlap by one sentence. This behavior of the chunker can reduce the significance of possible text segmentation errors.


License
-------

The **Smart-Chunker** (`smart-chunker`) is Apache 2.0 - licensed.