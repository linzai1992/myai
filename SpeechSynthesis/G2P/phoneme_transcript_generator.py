import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from g2p_model import G2PModel
from g2p_data_batcher import G2PDataBatcher
import tensorflow as tf

class PhonemeTranscriptGenerator:
    def __init__(self, root_dir_path, model_checkpoint_path):
        self.batcher = G2PDataBatcher("data/cmudict_proc.txt")
        with tf.Session() as session:
            print("Building model...")
            model = G2PModel(self.batcher.sequence_length, len(self.batcher.word_character_map), len(self.batcher.phoneme_map))
            saver = tf.train.Saver()

            print("Loading model checkpoint...")
            ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Failed loading model!")

            dirs = self.__find_data_dirs(root_dir_path)
            for data_dir in dirs:
                for dirpath, dirnames, filenames in os.walk(data_dir):
                    transcripts = list(filter(lambda x: x.endswith(".trans.txt"), filenames))
                    if len(transcripts) == 1:
                        transcript = transcripts[0]
                        print("Encoding transcript {}".format(transcript))
                        final_output = ""
                        with open(os.path.join(dirpath, transcript), "r") as f:
                            for line in f:
                                tokens = line.split(" ")
                                transcript_id = tokens[0]
                                words = tokens[1:]
                                if len(words) <= self.batcher.sequence_length:
                                    phon_sentence = ""
                                    for word in words:
                                        word = word.strip().lower()
                                        p = model.predict(session, self.batcher.batch_input_string(word))
                                        for i in range(p.shape[0]):
                                            phon_word = ""
                                            for j in range(self.batcher.sequence_length):
                                                phon = self.batcher.phoneme_map_inverse[p[i][j]]
                                                if phon == "<END>":
                                                    phon_word = phon_word.strip()
                                                    # phon_sentence += phon_word + " "
                                                    break
                                                phon_word += phon + " "
                                            phon_sentence += phon_word + " "
                                    final_output = "{}{} {}<END>\n".format(final_output, transcript_id, phon_sentence)
                        with open(os.path.join(dirpath, "phon_transcript.txt"), "w") as ofile:
                            ofile.write(final_output)
                        break


    def __find_data_dirs(self, root_path):
        dirs = []
        self.__find_data_dirs_rec(root_path, dirs)
        return dirs

    def __find_data_dirs_rec(self, root_path, dir_list):
        for (dirpath, dirnames, filenames) in os.walk(root_path):
            if len(filenames) > 0:
                flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
                if len(flacs) > 0:
                    dir_list.append(root_path)
            for dirname in dirnames:
                path = os.path.join(dirpath, dirname)
                self.__find_data_dirs_rec(path, dir_list)
            break

phon_gen = PhonemeTranscriptGenerator("..\\..\\SpeechSynthesisV2\\data\\test-clean\\LibriSpeech\\test-clean", "checkpoints")
