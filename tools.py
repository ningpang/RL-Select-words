import json
from argparse import ArgumentParser
from config import Config
parser = ArgumentParser(description='Relation Classification')
parser.add_argument('--config', default='config.ini')
args = parser.parse_args()
config = Config(args.config)

def read_relation(file):
    id2rel = json.load(open(file, 'r'), encoding='utf-8')
    rel2id = {}
    for i, rel in enumerate(id2rel):
        rel2id[rel] = i
    return id2rel, rel2id

id2rel, rel2id = read_relation(config.relation_file)
print(len(id2rel))