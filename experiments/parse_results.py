from glob import glob
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_mpi_type(file):
    return file.split('_')[-1].split('.')[0]


def extract_gpu_type(file):
    return file.split('_')[-2]


def extract_framework(file):
    return file.split('_')[0]


def extract_images_per_second(data):
    def _extract(line_string):
        if 'Total images/sec: ' in line_string:
            return float(line_string.split(':')[-1].strip())

    return np.array(list(
        filter(None,
               map(_extract, data))
    )).mean()


def extract_total_duration(data):
    def _extract(line_string):
        if 'Total duration: ' in line_string:
            return float(line_string.split(':')[-1].strip())

    return np.array(list(
        filter(None,
               map(_extract, data))
    )).mean()


def extact_data_length(data):
    for line in data:
        if 'Data length: ' in line:
            return int(line.split(':')[-1].strip())


def extract_batch_size(data):
    for line in data:
        if 'Batch size: ' in line:
            return int(line.split(':')[-1].strip().split(' ')[-1].strip(')'))


def extact_dataset(data):
    for line in data:
        if 'Dataset: ' in line:
            return line.split(':')[-1].strip()


def extract_num_devices(data):
    for line in data:
        if 'Num GPUs: ' in line:
            return int(float(line.split(': ')[-1].strip()))


extraction_funcs = {
    'Images/Second': extract_images_per_second,
    'Batch Size': extract_batch_size,
    'Data Length': extact_data_length,
    'Total Duration': extract_total_duration,
    'Dataset': extact_dataset,
    'GPUs': extract_num_devices,
}


def parse_results(file):
    logger.info('Processing {}'.format(file))
    with open(file) as f:
        data = f.readlines()
    results_dict = {key: func(data) for key, func in extraction_funcs.items()}
    results_dict['MPI'] = extract_mpi_type(file)
    results_dict['GPU Type'] = extract_gpu_type(file)
    results_dict['Framework'] = extract_framework(file)
    return results_dict


def write_json_to_file(json_dict, filename):
    """ Simple function to write JSON dictionaries to files
    """
    with open(filename, 'w') as outfile:
        json.dump(json_dict, outfile)


def main(path='*.results', filename='results.json'):
    logger.info('Reading files from {} and writing to {}'.format(path, filename))
    files = glob('*.results')
    logger.info('Found {} files'.format(len(files)))
    results = [parse_results(file) for file in files]
    logger.info('Writing results to {}'.format(filename))
    write_json_to_file(results, filename)


if __name__ == '__main__':
    main()
