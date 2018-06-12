import helper

en_path = 'data/corpus.en_ru.1m.en'
ru_path = 'data/corpus.en_ru.1m.ru'

en_text = helper.load_data(en_path)
ru_text = helper.load_data(ru_path)

en_text = en_text.lower()
ru_text = ru_text.lower()

en_sentences = en_text.split('\n')
ru_sentences = ru_text.split('\n')

assert len(en_sentences) == len(ru_sentences)
keep_percentage = 5
keep_sentences = len(en_sentences) * keep_percentage // 100

en_sentences = en_sentences[:keep_sentences]
ru_sentences = ru_sentences[:keep_sentences]

en_text = "\n".join(en_sentences)
ru_text = "\n".join(ru_sentences)

with open('data/corpus.en_ru.en.processed', 'w', encoding='utf-8') as out_file:
    out_file.write(en_text)

with open('data/corpus.en_ru.ru.processed', 'w', encoding='utf-8') as out_file:
    out_file.write(ru_text)
