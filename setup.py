from setuptools import setup, find_packages

import smart_chunker


long_description = '''
Smart-Chunker
===============

This smart chunker is a semantic chunker to prepare a
long document for retrieval augmented generation (RAG).

Unlike a usual chunker, it does not split the text into
identical groups of N tokens. Instead, it uses a cross-encoder
to calculate the similarity function between neighboring
sentences and divides the text based on the most significant
boundaries of semantic transitions, i.e. minima in the
above-mentioned similarity function.

The BAAI/bge-reranker-v2-m3, or any other model that supports the
AutoModelForSequenceClassification interface, should be used
as a cross encoder.

The smart chunker supports Russian and English.
'''

setup(
    name='smart-chunker',
    version=smart_chunker.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Smart-Chunker is a semantic chunker to prepare a long document for RAG',
    long_description=long_description,
    url='https://github.com/bond005/smart_chunker',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords=['smart-chunker', 'rag', 'chunker', 'cross-encoder', 'encoder', 'reranker'],
    install_requires=['nltk', 'razdel==0.5.0', 'sentencepiece', 'torch>=2.0.1', 'transformers>=4.38.1', 'accelerate==1.12.0'],
    test_suite='tests'
)
