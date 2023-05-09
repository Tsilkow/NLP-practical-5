from create_delex_data import *
import copy
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm

from utils import db_pointer, delexicalize
from utils.nlp import normalize

np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 40
_DOMAIN = "restaurant"



def create_prompts():
    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepare_slot_values_independent()
    delex_data = {}

    fin1 = open('data/woz2/data.json')
    data = json.load(fin1)

    counter = 0
    for dialogue_name in tqdm(data):
        if 'WOZ' not in dialogue_name:
            continue
        dialogue = data[dialogue_name]
        #print dialogue_name

        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])

            words = sent.split()
            sent = delexicalize.delexicalise(' '.join(words), dic)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            if counter <= 10:
                if idx % 2 == 1: prefix = 'System'
                else: prefix = 'Tourist'
                print(prefix, ': ', turn['text'], sep='')
                prefix += '_delex'
                print(prefix, ': ', sent, sep='')

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = add_db_pointer(turn)

                #print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

        delex_data[dialogue_name] = dialogue
        counter += 1


def create_entries():
    db = open('db/restaurant_db.json')
    json_parse = json.load(db)
    
    for restaurant in json_parse:
        for key in ['name', 'area', 'food', 'pricerange']:
            if key in restaurant: print(restaurant[key], end='')
            else: print('None')
            if key != 'pricerange': print(', ', end='')
        print()
