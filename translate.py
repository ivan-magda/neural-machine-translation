def getopts(argv):
    """Collect command-line options in a dictionary"""
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    lowercase = sentence.lower()

    default = vocab_to_int['<UNK>']
    sentence_id = [vocab_to_int.get(word, default) for word in lowercase.split()]

    return sentence_id


def translate_from_en_to_fr(translate_sentence):
    import tensorflow as tf
    import helper

    _, (source_vocab_to_int, target_vocab_to_int), \
    (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
    load_path = helper.load_params()

    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        batch_size = 1024

        translate_logits = sess.run(
            logits,
            {
                input_data: [translate_sentence] * batch_size,
                target_sequence_length: [len(translate_sentence) * 2] * batch_size,
                source_sequence_length: [len(translate_sentence)] * batch_size,
                keep_prob: 1.0
            }
        )[0]

    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))


if __name__ == '__main__':
    from sys import argv

    args = getopts(argv)

    if '--en' in args:
        translate_from_en_to_fr(args['--en'])
