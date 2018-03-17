import sys
import os
import re
sys.path.append('./analysis')
import pipe_to_model as pipe


in_root = 'raw_data'


def walk_and_process_audio(path,all_files = []):
    in_dir = os.path.join(in_root, path)
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        if os.path.isdir(filepath):
            walk_and_process_audio(os.path.join(path, file), all_files = all_files)
        elif os.path.isfile(filepath) and filepath.endswith('.wav'):
            in_audio = filepath
            try:
                print(filepath)
                all_files.append(filepath)
            except:
                print(filepath)
                next
    return all_files


all_files = walk_and_process_audio('')

def sound_type(file_name):
    PS = re.search('PS', file_name)
    if PS:
        return PS.group(0)
    BS = re.search('BS', file_name)
    if BS:
        return BS.group(0)
    TF = re.search('TF', file_name)
    if TF:
        return TF.group(0)


sound_type = [sound_type(x) for x in all_files]

data = pipe.get_frames_and_models(257,
                                  60000,
                                  all_files[0:200],
                                  sound_type[0:200],
                                  color_dictionary = {"PS":"red","BS":"blue","TF":"green"},
                                  perc_training = .8,
                                  directory = "./analysis/"
                                  )

