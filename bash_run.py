import os
import re
import time
import subprocess
import argparse

def os_command(cmd, env=None, stdout=subprocess.PIPE):
	with subprocess.Popen(cmd.split(), stdout=stdout, stderr=subprocess.STDOUT, encoding='latin-1', env=env) as p:
			(out,err) = p.communicate()
	return out

def eval_gpus():
	smi_re = re.compile('([0-9]+)MiB / ([0-9]+)MiB')
	smi_str = os_command('nvidia-smi')
	res = smi_re.findall(smi_str)
	if len(res) == 0:
			raise Exception('nvidia-smi not found!')
	gpu_mem = [int(d[0]) for d in res]
	print(gpu_mem)
	return gpu_mem

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-n', '--need', dest='need', required=True, help='gpu need.')
	args = arg_parser.parse_args()
	gpu_need = int(args.need)
	while True:
		free_gpus = [i for i, mem in enumerate(eval_gpus()) if mem < 500]
		if len(free_gpus) >= gpu_need:
			break
		time.sleep(300)

	cuda_str = ''.join(map(str, free_gpus[:gpu_need]))
	print(cuda_str)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_str) # "0, 1" for multiple	
	run_command = f'python run_ganist.py -l logs_temp -e 10000 -s 0 -g {cuda_str}'
	run_command = f'python run_training.py --num-gpus=4 --data-dir=/dresden/users/mk1391/evl/Data/lsun_bedroom_shift_tfr --config=config-e --dataset=lsun-bedroom-100k --mirror-augment=false --total-kimg 10000 --result-dir=results_sg_small_sbedroom128cc_1'
	os_command(run_command, stdout=None)
