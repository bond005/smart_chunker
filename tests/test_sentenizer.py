import os
import sys
from typing import List
import unittest

import nltk
import razdel
from transformers import AutoTokenizer


try:
    from smart_chunker.sentenizer import split_text_into_sentences, split_sentence
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from smart_chunker.sentenizer import split_text_into_sentences, split_sentence


def word_tokenize_en(s: str) -> List[str]:
    return nltk.wordpunct_tokenize(s)


def word_tokenize_ru(s: str) -> List[str]:
    return [it.text for it in razdel.tokenize(s)]


def sent_tokenize_en(s: str) -> List[str]:
    return nltk.sent_tokenize(s)


def sent_tokenize_ru(s: str) -> List[str]:
    return [it.text for it in razdel.sentenize(s)]


class TestSentenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_local_path = os.path.join(os.path.dirname(__file__), 'testdata', 'bge_tokenizer')
        cls.tokenizer = AutoTokenizer.from_pretrained(model_local_path)

    def test_split_sentence_ru_01(self):
        source_text = ('Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
                       'профессионалов — помогают им противостоять ветру, осадкам, холоду. '
                       'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет '
                       'сниженного веса, улучшенной эргономики и надежности. Со дня основания и до сегодняшнего дня '
                       'в Сплаве работают люди, небезразличные к туризму и другим видам активного отдыха на природе. '
                       'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год '
                       'активно ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта. '
                       'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых '
                       'экспедиций. А еще они пишут нам статьи и дают интервью в Блоге СПЛАВ. Неоценима и '
                       'помощь покупателей, неизменно присылающих обратную связь через сотни историй о '
                       'реальном использовании. Эти данные неизменно ложатся в основу новых разработок. '
                       'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась. '
                       'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, '
                       'приглашаем экспертов, проводим лекции, поддерживаем туристические мероприятия и '
                       'спортивные федерации. Мы невероятно рады тому, что количество наших клиентов неизменно растет. '
                       'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться '
                       'своим мнением с нами. Следуя своему девизу, компания “Сплав” открывает новые пространства и '
                       'помогает делать собственные открытия своим клиентам и партнерам.')
        sentences = split_sentence(long_sentence=source_text, max_seq_len=20, llm_tokenizer=self.tokenizer,
                                   word_tokenizer=word_tokenize_ru)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 1)
        for idx, val in enumerate(sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertGreater(len(val), 0, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, val.strip(), msg=f'Sentence {idx} is wrong: {val}')
        self.assertEqual(''.join(source_text.split()).strip(), ''.join(' '.join(sentences).strip().split()),
                         msg='\n' + ' '.join(sentences).strip())

    def test_split_sentence_ru_02(self):
        source_text = ('Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
                       'профессионалов — помогают им противостоять ветру, осадкам, холоду.')
        sentences = split_sentence(long_sentence=source_text, max_seq_len=100, llm_tokenizer=self.tokenizer,
                                   word_tokenizer=word_tokenize_ru)
        self.assertIsInstance(sentences, list)
        self.assertEqual(len(sentences), 1)
        for idx, val in enumerate(sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertGreater(len(val), 0, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, val.strip(), msg=f'Sentence {idx} is wrong: {val}')
        self.assertEqual(' '.join(source_text.split()).strip(), ' '.join(sentences).strip())

    def test_split_sentence_en_01(self):
        source_text = ('Founded in 1947 in Annecy, in the heart of the French Alps, Salomon has always been driven '
                       'by a passion for outdoor sports. In 1957, the brand revolutionized skiing with the launch of '
                       'the innovative “Skade” toe piece for ski bindings, setting the foundation for '
                       'future innovations. In 1979, Salomon introduced the SX90 ski boot, which became a landmark in '
                       'ski footwear. A year later, the SNS Nordic Boot and Binding System cemented '
                       'Salomon\'s leadership in winter sports. By 1991, the Monocoque ski design made Salomon '
                       'the first full-system ski brand, transforming the skiing experience. In 2019, Salomon opened '
                       'the Advanced Shoe Factory 4.0, a cutting-edge robotic production facility in France, '
                       'underscoring its focus on innovation and sustainability. Celebrating its 75th anniversary '
                       'in 2022, Salomon had fully evolved into a leading footwear brand. In 2024, the opening of '
                       'a flagship store on the Champs-Élysées in Paris marked a new chapter, just in time for '
                       'the Olympic Games.')
        sentences = split_sentence(long_sentence=source_text, max_seq_len=15, llm_tokenizer=self.tokenizer,
                                   word_tokenizer=word_tokenize_en)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 1)
        for idx, val in enumerate(sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertGreater(len(val), 0, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, val.strip(), msg=f'Sentence {idx} is wrong: {val}')
        self.assertEqual(''.join(source_text.split()).strip(), ''.join(' '.join(sentences).strip().split()),
                         msg='\n' + ' '.join(sentences).strip())

    def test_split_sentence_en_02(self):
        source_text = ('Founded in 1947 in Annecy, in the heart of the French Alps, Salomon has always been driven '
                       'by a passion for outdoor sports.')
        sentences = split_sentence(long_sentence=source_text, max_seq_len=100, llm_tokenizer=self.tokenizer,
                                   word_tokenizer=word_tokenize_en)
        self.assertIsInstance(sentences, list)
        self.assertEqual(len(sentences), 1)
        for idx, val in enumerate(sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertGreater(len(val), 0, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, val.strip(), msg=f'Sentence {idx} is wrong: {val}')
        self.assertEqual(' '.join(source_text.split()).strip(), ' '.join(sentences).strip())

    def test_split_text_into_sentences_ru_01(self):
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
        true_sentences = [
            'Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
            'профессионалов — помогают им противостоять ветру, осадкам, холоду.',  # 0
            'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет сниженного веса,',  # 1
            'улучшенной эргономики и надежности.',  # 2
            'Со дня основания и до сегодняшнего дня в Сплаве работают люди, небезразличные к туризму и другим видам '
            'активного отдыха на природе.',  # 3
            'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год активно '
            'ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта.',  # 4
            'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых экспедиций.',  # 5
            'А еще они пишут нам статьи и дают интервью в Блоге СПЛАВ.',  # 6
            'Неоценима и помощь покупателей, неизменно присылающих обратную связь через сотни историй о реальном '
            'использовании.',  # 7
            'Эти данные неизменно ложатся в основу новых разработок.',  # 8
            'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась.',  # 9
            'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, приглашаем экспертов, '
            'проводим лекции, поддерживаем туристические мероприятия и спортивные федерации.',   # 10
            'Мы невероятно рады тому, что количество наших клиентов неизменно растет.',  # 11
            'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться своим '
            'мнением с нами.',  # 12
            'Следуя своему девизу, компания “Сплав” открывает новые пространства и помогает делать собственные '
            'открытия своим клиентам и партнерам.',  # 13
        ]
        calculated_sentences = split_text_into_sentences(
            source_text=source_text, newline_as_separator=True,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(true_sentences), len(calculated_sentences))
        for idx, val in enumerate(calculated_sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, true_sentences[idx], msg=f'Sentence {idx} is wrong.')

    def test_split_text_into_sentences_ru_02(self):
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
        true_sentences = [
            'Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
            'профессионалов — помогают им противостоять ветру, осадкам, холоду.',  # 0
            'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет сниженного веса, '
            'улучшенной эргономики и надежности.',  # 1
            'Со дня основания и до сегодняшнего дня в Сплаве работают люди, небезразличные к туризму и другим видам '
            'активного отдыха на природе.',  # 2
            'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год активно '
            'ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта.',  # 3
            'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых экспедиций.',  # 4
            'А еще они пишут нам статьи и дают интервью в Блоге СПЛАВ.',  # 5
            'Неоценима и помощь покупателей, неизменно присылающих обратную связь через сотни историй о реальном '
            'использовании.',  # 6
            'Эти данные неизменно ложатся в основу новых разработок.',  # 7
            'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась.',  # 8
            'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, приглашаем экспертов, '
            'проводим лекции, поддерживаем туристические мероприятия и спортивные федерации.',   # 9
            'Мы невероятно рады тому, что количество наших клиентов неизменно растет.',  # 10
            'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться своим '
            'мнением с нами.',  # 11
            'Следуя своему девизу, компания “Сплав” открывает новые пространства и помогает делать собственные '
            'открытия своим клиентам и партнерам.',  # 12
        ]
        calculated_sentences = split_text_into_sentences(
            source_text=source_text, newline_as_separator=False,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(true_sentences), len(calculated_sentences))
        for idx, val in enumerate(calculated_sentences):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, true_sentences[idx], msg=f'Sentence {idx} is wrong.')

    def test_split_text_into_sentences_ru_03(self):
        calculated_sentences = split_text_into_sentences(
            source_text='', newline_as_separator=True,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(calculated_sentences), 0)

    def test_split_text_into_sentences_ru_04(self):
        calculated_sentences = split_text_into_sentences(
            source_text='', newline_as_separator=False,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(calculated_sentences), 0)

    def test_split_text_into_sentences_ru_05(self):
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
        true_sentences = [
            'Мы создаем одежду и снаряжение, которые расширяют возможности путешественников и вооруженных '
            'профессионалов — помогают им противостоять ветру, осадкам, холоду.',  # 0
            'Способствуют тому, чтобы меньше уставать в пути и быстрее восстанавливаться за счет сниженного веса, '
            'улучшенной эргономики и надежности.',  # 1
            'Со дня основания и до сегодняшнего дня в Сплаве работают люди, небезразличные к туризму и другим видам '
            'активного отдыха на природе.',  # 2
            'Разработчики, конструктора, торговый персонал и многие другие сотрудники сами уже не один год активно '
            'ходят в походы, сплавляются по рекам, занимаются ски-туром и другими видами спорта.',  # 3
            'Нам помогают эксперты-путешественники, тестирующие экипировку СПЛАВ в условиях непростых экспедиций.',  # 4
            'А еще они пишут нам статьи и дают интервью в Блоге СПЛАВ.',  # 5
            'Неоценима и помощь покупателей, неизменно присылающих обратную связь через сотни историй о реальном '
            'использовании.',  # 6
            'Эти данные неизменно ложатся в основу новых разработок.',  # 7
            'Компания стремится к тому, чтобы культура туризма и путешествий в России жила и развивалась.',  # 8
            'Мы делимся своими знаниями и опытом через статьи, видео и другие медиа-форматы, приглашаем экспертов, '
            'проводим лекции, поддерживаем туристические мероприятия и спортивные федерации.',   # 9
            'Мы невероятно рады тому, что количество наших клиентов неизменно растет.',  # 10
            'Признательны им за доверие бренду “Сплав”, за рекомендации друзьям, за готовность делиться своим '
            'мнением с нами.',  # 11
            'Следуя своему девизу, компания “Сплав” открывает новые пространства и помогает делать собственные '
            'открытия своим клиентам и партнерам.',  # 12
        ]
        calculated_sentences_1 = split_text_into_sentences(
            source_text=source_text, newline_as_separator=False,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences_1, list)
        self.assertEqual(len(true_sentences), len(calculated_sentences_1))
        for idx, val in enumerate(calculated_sentences_1):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, true_sentences[idx], msg=f'Sentence {idx} is wrong.')
        calculated_sentences_2 = split_text_into_sentences(
            source_text=source_text, newline_as_separator=False,
            max_seq_len=15, llm_tokenizer=self.tokenizer,
            sent_tokenizer=sent_tokenize_ru, word_tokenizer=word_tokenize_ru
        )
        self.assertIsInstance(calculated_sentences_2, list)
        self.assertLess(len(true_sentences), len(calculated_sentences_2))
        for idx, val in enumerate(calculated_sentences_2):
            self.assertIsInstance(val, str, msg=f'Sentence {idx} is wrong: {val}')
            self.assertGreater(len(val), 0, msg=f'Sentence {idx} is wrong: {val}')
            self.assertEqual(val, val.strip(), msg=f'Sentence {idx} is wrong: {val}')
        self.assertEqual(''.join(source_text.split()).strip(),
                         ''.join(' '.join(calculated_sentences_2).strip().split()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
