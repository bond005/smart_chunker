import sys
import os
import unittest
import warnings

import torch

try:
    from smart_chunker.chunker import SmartChunker
    from smart_chunker.sentenizer import split_text_into_sentences
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from smart_chunker.chunker import SmartChunker
    from smart_chunker.sentenizer import split_text_into_sentences


class TestSmartChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path_to_model = os.path.join(os.path.dirname(__file__), 'testdata', 'bge_reranker')
        if not os.path.isfile(os.path.join(cls.path_to_model, 'model.safetensors')):
            warnings.warn(f'There are no model files in the directory "{cls.path_to_model}".')
            cls.path_to_model = 'BAAI/bge-reranker-v2-m3'

    def test_chunker_ru_1(self):
        source_text = ('Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
                       'профессионалов — помогают им противостоять ветру, осадкам, холоду. '
                       'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет '
                       'сниженного веса,\nулучшенной эргономики и надежности. Со дня основания и до сегодняшнего дня '
                       'в Сплаве работают люди, небезразличные к туризму и другим видам активного отдыха на природе. '
                       'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год '
                       'активно ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта. '
                       'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых '
                       'экспедиций.\n\nА еще они пишут нам статьи и дают интервью в Блоге СПЛАВ. Неоценима и '
                       'помощь покупателей, неизменно присылающих обратную связь через сотни историй о '
                       'реальном использовании. Эти данные неизменно ложатся в основу новых разработок. '
                       'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась. '
                       'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, '
                       'приглашаем экспертов, проводим лекции, поддерживаем туристические мероприятия и '
                       'спортивные федерации. Мы невероятно рады тому, что количество наших клиентов неизменно растет. '
                       'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться '
                       'своим мнением с нами. Следуя своему девизу, компания “Сплав” открывает новые пространства и '
                       'помогает делать собственные открытия своим клиентам и партнерам.')
        chunker = SmartChunker(reranker_name=self.path_to_model, newline_as_separator=True,
                               device='cuda:0' if torch.cuda.is_available() else 'cpu',
                               max_chunk_length=50, minibatch_size=4, verbose=True)
        sentences = split_text_into_sentences(source_text, newline_as_separator=chunker.newline_as_separator,
                                              lang=chunker.language, max_seq_len=(2 * chunker.max_chunk_length) // 3,
                                              tokenizer=chunker.tokenizer_)
        chunks = chunker.split_into_chunks(source_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 1)
        self.assertLess(len(chunks), len(sentences))
        prev_pos = 0
        normalized_text = ' '.join(source_text.lower().split())
        coverage_of_normalized_text = [(1 if normalized_text[idx].isspace() else 0)
                                       for idx in range(len(normalized_text))]
        set_of_normalized_chunks = set()
        for chunk_idx, cur_chunk in enumerate(chunks):
            normalized_chunk = ' '.join(cur_chunk.lower().split())
            self.assertGreater(len(normalized_chunk), 0,
                               msg=f'Chunk {chunk_idx} is bad!\n{" ".join(cur_chunk.split())}')
            set_of_normalized_chunks.add(normalized_chunk)
            found_idx = normalized_text[prev_pos:].find(normalized_chunk)
            self.assertGreaterEqual(found_idx, 0, msg=f'Chunk {chunk_idx} is bad!\n{" ".join(cur_chunk.split())}')
            chunk_start = prev_pos + found_idx
            chunk_end = chunk_start + len(normalized_chunk)
            self.assertFalse(normalized_text[chunk_start].isspace())
            self.assertFalse(normalized_text[chunk_end - 1].isspace())
            for char_idx in range(chunk_start, chunk_end):
                coverage_of_normalized_text[char_idx] = 1
            prev_pos = chunk_start + 1
        self.assertEqual(set(coverage_of_normalized_text), {1})
        self.assertEqual(len(set_of_normalized_chunks), len(chunks))

    def test_chunker_ru_2(self):
        source_text = ('Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
                       'профессионалов — помогают им противостоять ветру, осадкам, холоду. '
                       'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет '
                       'сниженного веса,\nулучшенной эргономики и надежности. Со дня основания и до сегодняшнего дня '
                       'в Сплаве работают люди, небезразличные к туризму и другим видам активного отдыха на природе. '
                       'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год '
                       'активно ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта. '
                       'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых '
                       'экспедиций.\n\nА еще они пишут нам статьи и дают интервью в Блоге СПЛАВ. Неоценима и '
                       'помощь покупателей, неизменно присылающих обратную связь через сотни историй о '
                       'реальном использовании. Эти данные неизменно ложатся в основу новых разработок. '
                       'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась. '
                       'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, '
                       'приглашаем экспертов, проводим лекции, поддерживаем туристические мероприятия и '
                       'спортивные федерации. Мы невероятно рады тому, что количество наших клиентов неизменно растет. '
                       'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться '
                       'своим мнением с нами. Следуя своему девизу, компания “Сплав” открывает новые пространства и '
                       'помогает делать собственные открытия своим клиентам и партнерам.')
        chunker = SmartChunker(reranker_name=self.path_to_model, newline_as_separator=False,
                               device='cuda:0' if torch.cuda.is_available() else 'cpu',
                               max_chunk_length=50, minibatch_size=4, verbose=True)
        sentences = split_text_into_sentences(source_text, newline_as_separator=chunker.newline_as_separator,
                                              lang=chunker.language, max_seq_len=(2 * chunker.max_chunk_length) // 3,
                                              tokenizer=chunker.tokenizer_)
        chunks = chunker.split_into_chunks(source_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 1)
        self.assertLess(len(chunks), len(sentences))
        prev_pos = 0
        normalized_text = ' '.join(source_text.lower().split())
        coverage_of_normalized_text = [(1 if normalized_text[idx].isspace() else 0)
                                       for idx in range(len(normalized_text))]
        set_of_normalized_chunks = set()
        for chunk_idx, cur_chunk in enumerate(chunks):
            normalized_chunk = ' '.join(cur_chunk.lower().split())
            self.assertGreater(len(normalized_chunk), 0,
                               msg=f'Chunk {chunk_idx} is bad!\n{" ".join(cur_chunk.split())}')
            set_of_normalized_chunks.add(normalized_chunk)
            found_idx = normalized_text[prev_pos:].find(normalized_chunk)
            self.assertGreaterEqual(found_idx, 0, msg=f'Chunk {chunk_idx} is bad!\n{" ".join(cur_chunk.split())}')
            chunk_start = prev_pos + found_idx
            chunk_end = chunk_start + len(normalized_chunk)
            self.assertFalse(normalized_text[chunk_start].isspace())
            self.assertFalse(normalized_text[chunk_end - 1].isspace())
            for char_idx in range(chunk_start, chunk_end):
                coverage_of_normalized_text[char_idx] = 1
            prev_pos = chunk_start + 1
        self.assertEqual(set(coverage_of_normalized_text), {1})
        self.assertEqual(len(set_of_normalized_chunks), len(chunks))


if __name__ == '__main__':
    unittest.main(verbosity=2)

