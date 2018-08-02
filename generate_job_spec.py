import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
# Config for Intel
cmd_for_intel = \
    """source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh; 
    echo $AZ_BATCH_HOST_LIST; 
    mpirun -n {total_processes} -ppn {processes_per_node} {hosts} 
    -env I_MPI_FABRICS=dapl 
    -env I_MPI_DAPL_PROVIDER=ofa-v2-ib0 
    -env I_MPI_DYNAMIC_CONNECTION=0 
    -env I_MPI_DEBUG=6 
    -env I_MPI_HYDRA_DEBUG=on 
    -env DISTRIBUTED=True 
    {fake} 
    {fake_length} 
    python -u {script}""".replace('\n', '')

# Config for OpenMPI
cmd_for_openmpi = \
    """echo $AZ_BATCH_HOST_LIST; 
    cat $AZ_BATCHAI_MPI_HOST_FILE; 
    mpirun -np {total_processes} {hosts} 
    -bind-to none -map-by slot 
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH 
    -mca btl_tcp_if_include eth0 
    -x NCCL_SOCKET_IFNAME=eth0 
    -mca btl ^openib 
    -x NCCL_IB_DISABLE=1 
    -x DISTRIBUTED=True 
    -x AZ_BATCHAI_INPUT_TRAIN
    -x AZ_BATCHAI_INPUT_TEST
    {fake} 
    {fake_length} 
    --allow-run-as-root 
    python -u {script}""".replace('\n', '')

# Running on single node without mpi
cmd_local = """{fake} {fake_length} python -u {script}""".replace('\n', '')

cmd_choice_dict = {
    'openmpi': cmd_for_openmpi,
    'intelmpi': cmd_for_intel,
    'local': cmd_local
}

hosts_param = {
    'openmpi': '--hostfile $AZ_BATCHAI_MPI_HOST_FILE ',
    'intelmpi': '-hosts $AZ_BATCH_HOST_LIST ',
    'local': ''
}

fake_param = {
    'openmpi': '-x FAKE=True  ',
    'intelmpi': '-env FAKE=True ',
    'local': ' FAKE=True '
}

fake_length_param = {
    'openmpi': '-x FAKE_DATA_LENGTH={}  ',
    'intelmpi': '-env FAKE_DATA_LENGTH={} ',
    'local': ' FAKE_DATA_LENGTH={} '
}


def _hosts_for(mpitype, node_count):
    if node_count > 1:
        return hosts_param.get(mpitype, '')
    else:
        return hosts_param.get('local')


def _fake_for(mpitype, data):
    if data is None:
        return fake_param.get(mpitype, '')
    else:
        return ''


def _fake_length_for(mpitype, fake_length, data):
    if data is None:
        return fake_length_param.get(mpitype, '').format(fake_length)
    else:
        return ''


def _prepare_command(mpitype, total_processes, processes_per_node, script, node_count, data=None, synthetic_length=1281167):
    command = cmd_choice_dict.get(mpitype, cmd_for_intel)
    return command.format(total_processes=total_processes,
                          processes_per_node=processes_per_node,
                          script=script,
                          hosts=_hosts_for(mpitype, node_count),
                          fake=_fake_for(mpitype, data),
                          fake_length=_fake_length_for(mpitype, synthetic_length, data))


def append_data_paths(job_template_dict, data_path):
    job_template_dict['properties']['inputDirectories'].extend([{
        "id": "TRAIN",
        "path": data_path,
    },
        {
            "id": "TEST",
            "path": data_path,
        }])
    return job_template_dict


def generate_job_dict(image_name,
                      command,
                      node_count=2):
    return {
        "$schema": "https://raw.githubusercontent.com/Azure/BatchAI/master/schemas/2017-09-01-preview/job.json",
        "properties": {
            "nodeCount": node_count,
            "customToolkitSettings": {
                "commandLine": command
            },
            "stdOutErrPathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/extfs",
            "inputDirectories": [{
                "id": "SCRIPTS",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/extfs/scripts"
            },
            ],
            "outputDirectories": [{
                "id": "MODEL",
                "pathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/extfs",
                "pathSuffix": "Models"
            }],
            "containerSettings": {
                "imageSourceRegistry": {
                    "image": image_name
                }
            }
        }
    }


def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)
        outfile.write('\n\n')


def synthetic_data_job(image_name,
                       mpitype,
                       script,
                       filename='job.json',
                       node_count=2,
                       total_processes=None,
                       processes_per_node=4,
                       synthetic_length=1281167):
    logger.info('Creating manifest for job with synthetic data {} with {} image...'.format(filename, image_name))
    total_processes = processes_per_node * node_count if total_processes is None else total_processes
    command = _prepare_command(mpitype,
                               total_processes,
                               processes_per_node,
                               script,
                               node_count,
                               synthetic_length=synthetic_length)
    job_template = generate_job_dict(image_name,
                      command,
                      node_count=node_count)
    write_json_to_file(job_template, filename)
    logger.info('Done')


def imagenet_data_job(image_name,
                      mpitype,
                      script,
                      data_path,
                      filename='job.json',
                      node_count=2,
                      total_processes=None,
                      processes_per_node=4):
    logger.info('Creating manifest for job with real data {} with {} image...'.format(filename, image_name))
    total_processes = processes_per_node * node_count if total_processes is None else total_processes
    command = _prepare_command(mpitype,
                               total_processes,
                               processes_per_node,
                               script,
                               node_count,
                               data=data_path)
    job_template = generate_job_dict(image_name,
                                     command,
                                     node_count=node_count)
    job_template = append_data_paths(job_template, data_path)
    write_json_to_file(job_template, filename)
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest')
    parser.add_argument('docker_image', type=str,
                        help='docker image to use')
    parser.add_argument('mpi', type=str,
                        help='mpi to use, must be install in the docker image provided options:[intelmpi, openmpi, local]')
    parser.add_argument('script', type=str,
                        help='script to run')
    parser.add_argument('--filename', '-f', dest='filename', type=str, nargs='?',
                        default='job.json',
                        help='name of the file to save job spec to')
    parser.add_argument('--node_count', '-n', dest='node_count', type=int, nargs='?',
                        default=1, help='the number of nodes to run the job across')
    parser.add_argument('--ppn', dest='processes_per_node', type=int, nargs='?',
                        default=4,
                        help='number of GPU proceses to run per node')
    parser.add_argument('--data', dest='data', type=str, nargs='?',
                        default=None,
                        help='the path where the imagenet data is stored')
    parser.add_argument('--synthetic_length', '-l', dest='synthetic_length', type=str, nargs='?',
                        default=1281167,
                        help='the length of the fake data [default=size of imagenet 1281167]')
    args = parser.parse_args()
    if args.data is None:
        synthetic_data_job(args.docker_image,
                           args.mpi,
                           args.script,
                           filename=args.filename,
                           node_count=args.node_count,
                           processes_per_node=args.processes_per_node,
                           synthetic_length=args.synthetic_length)
    else:
        imagenet_data_job(args.docker_image,
                           args.mpi,
                           args.script,
                           args.data,
                           filename=args.filename,
                           node_count=args.node_count,
                           processes_per_node=args.processes_per_node)
