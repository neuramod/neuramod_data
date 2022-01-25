import openpyxl
import subprocess
import glob
import os
import sys
#import h5py

xl_file = "D:\\Dropbox (Personal)\\XXXX_PERS_Git\\neuramod_data\\data\\raw\\stim\\experiment_1\\lastSelectedTokens.xlsx"
wb_obj = openpyxl.load_workbook(xl_file)
sheet = wb_obj.active

dataset_base_path = "D:\\Dropbox (Personal)\\XXXX_PERS_Git\\neuramod_data\\data\\raw\\stim\\experiment_1"
generator_path = "D:\\Dropbox (Personal)\\XXXX_PERS_Git\\neuramod_experiments\\experiment_1\\generator.py"
token_num = "12"
step_num = "None"
res_x = "3840"
res_y = "2160"
format_tok = "tif"

for i, row in enumerate(sheet.iter_rows(values_only=True)):
	if i>0:
		token = row[1]

		prior = "{}_".format(token.rsplit('_', 2)[0])
		cmd = ['python',
       			generator_path ,
       			dataset_base_path ,
       			'pull',
       			prior,
       			dataset_base_path ,
       			token_num,
       			step_num,
       			res_x,
       			res_y,
       			format_tok]
		subprocess.check_call(cmd,stdout=sys.stdout,stderr=subprocess.STDOUT)
		
		generated_tokens = glob.glob('{}*.{}'.format(prior,format_tok))
		for gen_ in generated_tokens:
			if token not in gen_:
				os.remove(gen_)
		