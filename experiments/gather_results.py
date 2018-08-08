import json
import logging
from glob import iglob
from itertools import chain
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_storage_type(json_data, filename):
    json_data['Storage Type'] = os.path.dirname(filename)


def read_json(filename):
    logger.info('Reading {}...'.format(filename))
    with open(filename) as f:
        return json.load(f)


def write_json_to_file(json_data, filename):
    with open(filename, 'w') as outfile:
        json.dump(json_data, outfile)


def main(filename='all_results.json'):
    files = iglob('**/results.json', recursive=True)
    json_data = (read_json(i) for i in files)
    augmented_json_data = (append_storage_type(j, f) for j, f in zip(json_data, files))
    write_json_to_file(list(chain.from_iterable(augmented_json_data)), filename)
    logger.info('All results written to  {}'.format(filename))


if __name__ == "__main__":
    main()
