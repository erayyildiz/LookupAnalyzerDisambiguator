# coding=utf-8
import dynet as dy
import numpy as np
import pickle
from datetime import datetime

from candidate_generators import TurkishStemSuffixCandidateGenerator
from data_utils import data_generator
import logging.config

from utils import to_lower, WordStruct

logging.config.fileConfig('resources/logging.ini')
logger = logging.getLogger(__file__)


class AnalysisScorerModel(object):
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    @classmethod
    def _create_vocab_chars(cls, sentences):
        char2id = dict()
        char2id["**Unknown"] = 0
        char2id["<"] = len(char2id) + 1
        char2id["/"] = len(char2id) + 1
        char2id[">"] = len(char2id) + 1
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id) + 1
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id:
                            char2id[ch] = len(char2id) + 1
        return char2id

    @classmethod
    def _create_vocab_tags(cls, sentences):
        tag2id = dict()
        tag2id["**Unknown"] = 0
        for sentence in sentences:
            for word in sentence:
                for tags in word.tags:
                    for tag in tags:
                        if tag not in tag2id:
                            tag2id[tag] = len(tag2id) + 1
        return tag2id

    @classmethod
    def _encode(cls, tokens, vocab):
        res = []
        for token in tokens:
            if token in vocab:
                res.append(vocab[token])
            else:
                res.append(0)
        return res

    @classmethod
    def _embed(cls, token, char_embedding_table):
        return [char_embedding_table[ch] for ch in token]

    def __init__(self, train_from_scratch=True, char_representation_len=128,
                 word_lstm_rep_len=256, train_data_path="data/data.train.txt",
                 dev_data_path="data/data.dev.txt", test_data_paths=["data/data.test.txt"],
                 model_file_name=None, case_sensitive=True):
        assert word_lstm_rep_len % 2 == 0
        self.case_sensitive = case_sensitive
        if train_from_scratch:
            assert train_data_path
            logger.info("Loading data...")
            self.train_data_path = train_data_path
            self.dev_data_path = dev_data_path
            self.test_data_paths = test_data_paths
            self.train = data_generator(train_data_path, add_gold_labels=True)
            logger.info("Creating or Loading Vocabulary...")
            self.char2id = self._create_vocab_chars(self.train)
            self.train = data_generator(train_data_path, add_gold_labels=True)
            self.tag2id = self._create_vocab_tags(self.train)
            self.dev = None
            self.tests = []
            for test_path in self.test_data_paths:
                self.tests.append(data_generator(test_path, add_gold_labels=True))
            logger.info("Building model...")
            self.model = dy.Model()
            self.trainer = dy.AdamTrainer(self.model)
            self.CHARS_LOOKUP = self.model.add_lookup_parameters((len(self.char2id) + 2, char_representation_len))
            self.TAGS_LOOKUP = self.model.add_lookup_parameters((len(self.tag2id) + 2, char_representation_len))
            self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.train_model(model_name=model_file_name)
        else:
            logger.info("Loading Pre-Trained Model")
            assert model_file_name
            self.load_model(model_file_name, char_representation_len, word_lstm_rep_len)

    def propogate(self, sentence):
        dy.renew_cg()
        fwd_rnn_surface_init = self.fwdRNN_surface.initial_state()
        bwd_rnn_surface_init = self.bwdRNN_surface.initial_state()
        fwd_rnn_root_init = self.fwdRNN_root.initial_state()
        bwd_rnn_root_init = self.bwdRNN_root.initial_state()
        fwd_rnn_tag_init = self.fwdRNN_tag.initial_state()
        bwd_rnn_tag_init = self.bwdRNN_tag.initial_state()
        fwd_rnn_context_init = self.fwdRNN_context.initial_state()
        bwd_rnn_context_init = self.bwdRNN_context.initial_state()

        # CONTEXT REPRESENTATIONS
        surface_words_rep = []
        for index, word in enumerate(sentence):
            encoded_surface_word = self._encode(word.surface_word, self.char2id)
            surface_word_char_embeddings = self._embed(encoded_surface_word, self.CHARS_LOOKUP)
            fw_exps_surface_word = fwd_rnn_surface_init.transduce(surface_word_char_embeddings)
            bw_exps_surface_word = bwd_rnn_surface_init.transduce(reversed(surface_word_char_embeddings))
            surface_word_rep = dy.concatenate([fw_exps_surface_word[-1], bw_exps_surface_word[-1]])
            surface_words_rep.append(surface_word_rep)
        fw_exps_context = fwd_rnn_context_init.transduce(surface_words_rep)
        bw_exps_context = bwd_rnn_context_init.transduce(reversed(surface_words_rep))
        scores = []
        # Stem and POS REPRESENTATIONS
        for index, word in enumerate(sentence):
            encoded_roots = [self._encode(root, self.char2id) for root in word.roots]
            encoded_tags = [self._encode(tag, self.tag2id) for tag in word.tags]
            roots_embeddings = [self._embed(root, self.CHARS_LOOKUP) for root in encoded_roots]
            tags_embeddings = [self._embed(tag, self.TAGS_LOOKUP) for tag in encoded_tags]
            analysis_representations = []
            for root_embedding, tag_embedding in zip(roots_embeddings, tags_embeddings):
                fw_exps_root = fwd_rnn_root_init.transduce(root_embedding)
                bw_exps_root = bwd_rnn_root_init.transduce(reversed(root_embedding))
                root_representation = dy.rectify(dy.concatenate([fw_exps_root[-1], bw_exps_root[-1]]))

                fw_exps_tag = fwd_rnn_tag_init.transduce(tag_embedding)
                bw_exps_tag = bwd_rnn_tag_init.transduce(reversed(tag_embedding))
                tag_representation = dy.rectify(dy.concatenate([fw_exps_tag[-1], bw_exps_tag[-1]]))

                analysis_representations.append(dy.rectify(dy.esum([root_representation, tag_representation])))

            left_context_rep = fw_exps_context[index]
            right_context_rep = bw_exps_context[len(sentence) - index - 1]
            context_rep = dy.tanh(dy.esum([left_context_rep, right_context_rep]))
            scores.append(
                (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * dy.concatenate(analysis_representations, 1))[
                    0])

        return scores

    def get_loss(self, sentence):
        scores = self.propogate(sentence)
        errs = []
        for i, score in enumerate(scores):
            root_err = dy.pickneglogsoftmax(score, 0)
            errs.append(root_err)

        return dy.esum(errs)

    def predict_indices(self, sentence):
        selected_indices = []
        scores = self.propogate(sentence)
        for score in scores:
            probs = dy.softmax(score)
            selected_indices.append(np.argmax(probs.npvalue()))
        return selected_indices

    def predict_probs(self, sentence):
        res = []
        scores = self.propogate(sentence)
        for score in scores:
            probs = dy.softmax(score)
            res.append(probs.npvalue())
        return res

    def predict(self, tokens):
        sentence = []
        candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=False)
        for token in tokens:
            token = to_lower(token)
            candidate_analyzes = candidate_generator.get_analysis_candidates(token)
            roots = []
            tags = []
            for analysis in candidate_analyzes:
                roots.append(analysis[0])
                tags.append(analysis[2])
            sentence.append(WordStruct(token, roots, [], tags))
        selected_indices = self.predict_indices(sentence)
        res = []
        for i, j in enumerate(selected_indices):
            selected_analysis = sentence[i].roots[j] + "+" + "+".join(sentence[i].tags[j])
            selected_analysis = selected_analysis.replace("+DB", "^DB")
            res.append(selected_analysis)
        return res

    def calculate_acc(self, sentences):
        corrects = 0
        non_ambigious_count = 0
        total = 0
        for sentence in sentences:
            predicted_label_indexes = self.predict_indices(sentence)
            corrects += predicted_label_indexes.count(0)
            non_ambigious_count += [1 for w in sentence if len(w.roots) == 1].count(1)
            total += len(sentence)
        return (corrects * 1.0 / total), ((corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count))

    def train_model(self, model_name="model", early_stop=False, num_epoch=200):
        logger.info("Starting training...")
        max_acc = 0.0
        epoch_loss = 0
        for epoch in range(num_epoch):
            logger.info("Loading data...")
            self.train = data_generator(self.train_data_path, add_gold_labels=True)
            self.dev = data_generator(self.dev_data_path, add_gold_labels=True)
            self.tests = []
            for test_path in self.test_data_paths:
                self.tests.append(data_generator(test_path, add_gold_labels=True))
            t1 = datetime.now()
            count = 0
            for i, sentence in enumerate(self.train, 1):
                loss_exp = self.get_loss(sentence)
                cur_loss = loss_exp.scalar_value()
                epoch_loss += cur_loss
                loss_exp.backward()
                self.trainer.update()
                if i % 100 == 0:  # logger.info status
                    t2 = datetime.now()
                    delta = t2 - t1
                    logger.info("\rloss = {}  /  {} instances finished in  {} seconds     "
                                .format(epoch_loss / (i * 1.0), i, delta.seconds))
                count = i
            t2 = datetime.now()
            delta = t2 - t1
            logger.info("epoch {} finished in {} minutes. loss = {}"
                        .format(epoch, delta.seconds / 60.0, epoch_loss / count * 1.0))
            logger.info("Calculating Accuracy on dev set")
            epoch_loss = 0
            acc, amb_acc = self.calculate_acc(self.dev)
            logger.info("Accuracy on dev set:{}\nAmbiguous accuracy on dev set:{} ".format(acc, amb_acc))
            if acc > max_acc:
                max_acc = acc
                logger.info("Max accuracy increased, saving model...")
                self.save_model(model_name)
            elif early_stop and max_acc > acc:
                logger.info("Max accuracy did not increase, early stopping!")
                break

            logger.info("Calculating Accuracy on test sets")
            for q in range(len(self.test_data_paths)):
                logger.info("Calculating Accuracy on test set: {}".format(self.test_data_paths[q]))
                acc, amb_acc = self.calculate_acc(self.tests[q])
                logger.info(" accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc))

    def save_model(self, model_name):
        self.model.save("resources/models/" + model_name + ".model")
        with open("resources/models/" + model_name + ".char2id", "wb") as f:
            pickle.dump(self.char2id, f)
        with open("resources/models/" + model_name + ".tag2id", "wb") as f:
            pickle.dump(self.tag2id, f)

    def load_model(self, model_name, char_representation_len, word_lstm_rep_len):
        with open("resources/models/" + model_name + ".char2id", "rb") as f:
            self.char2id = pickle.load(f)
        with open("resources/models/" + model_name + ".tag2id", "rb") as f:
            self.tag2id = pickle.load(f)

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((len(self.char2id) + 2, char_representation_len))
        self.TAGS_LOOKUP = self.model.add_lookup_parameters((len(self.tag2id) + 2, char_representation_len))
        self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.model.populate("resources/models/" + model_name + ".model")

    @staticmethod
    def create_from_existed_model(model_name):
        return AnalysisScorerModel(train_from_scratch=False, model_file_name=model_name)


def calculate_acc_on_testfile(file_path):
    model = AnalysisScorerModel.create_from_existed_model(model_name="lookup_disambiguator_wo_suffix")
    test_data = data_generator(file_path, add_gold_labels=True, case_sensitive=True)
    corrects = 0
    total = 0
    with open("data/incorrect_analyzes.csv", "w", encoding="UTF-8") as f:
        f.write("Surface\tGold\tPredicted\n")
        for sentence in test_data:
            predicted_indexes = model.predict_indices(sentence)
            for word, selected_index in zip(sentence, predicted_indexes):
                gold_analysis = word.roots[0] + "+" + "+".join(word.tags[0])
                gold_analysis = gold_analysis.replace("+DB", "^DB")
                selected_analysis = word.roots[selected_index] + "+" + "+".join(word.tags[selected_index])
                selected_analysis = selected_analysis.replace("+DB", "^DB")
                if to_lower(selected_analysis) == to_lower(gold_analysis):
                    corrects += 1
                else:
                    f.write("{}\t{}\t{}\n".format(word.surface_word, gold_analysis, selected_analysis))
                total += 1
        print("Accuracy: {}".format(corrects * 1.0 / total))


if __name__ == "__main__":
    AnalysisScorerModel(train_data_path="data/data.train.txt", dev_data_path="data/data.dev.txt",
                        test_data_paths=["data/test.merge", "data/data.test.txt", "Morph.Dis.Test.Hand.Labeled-20K.txt"],
                        model_file_name="lookup_disambiguator_wo_suffix", train_from_scratch=True)


