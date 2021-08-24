#lsl_stream.py
#
name			 = "lsl_stream"
version		  	 = "0.0.1" #
author		   	 = "Pierre Cutellic"
author_email	 = "cutellic@arch.ethz.ch"
home_page		 = "http://www.compmonks.com/"
license		  	 = "License" # License Type
description	  	 = "Provide lsl protocol to stream out data with user settings"

long_description = \
"""
	Provide lsl protocol to stream out data with user settings...

"""
#
####################################################################################################
# TODO + WARNINGS
# 
# 
####################################################################################################
# MODULES
import sys,os
sys.path.append(os.path.join(".","neurodesign_app","static","neurodesign_app","python"))
#sys.path.append(os.path.join("..",""))
#sys.path.append('./neurodesign_app/static/neurodesign_app/python') # include custom packages
#
try:
	from utilities.logging import xslog
	from pylsl import StreamInfo, StreamOutlet
except():
	print ("ERROR on script:{} version:{} description:{}".format(name,version,description))
	print(sys.exc_info())
	sys.exit(1)
#	
# VERSION CHECKS
# print('Python: {}'.format(sys.version))
# print('django: {}'.format(django.__version__))
#
####################################################################################################
# GLOBAL VARIABLES
#
#
####################################################################################################
#
# CLASSES___________________________________________________________________________________________
class Stream():
	""" LSL output stream class."""
	#
	# CONSTRUCTOR___________________________________________________________________________________
	def __init__(self, the_device = None, the_name = None, the_stream = None, the_channels = None, 
					   the_sample_rate = None):
		"""Configure and initialize a lsl outlet."""

		self.the_chunk_num = 12
		# DEBUG ____________________________________________________________________________________
		if the_sample_rate != None:
			info = StreamInfo(name = "{}".format(the_name), 
						  type = "{}".format(the_stream.output_datatype), 
						  channel_count = "{}".format(len(the_channels)), 
						  nominal_srate = the_sample_rate, 
						  channel_format = "float32",
						  source_id = "{}".format(the_device))
		else:
			info = StreamInfo(name = "{}".format(the_name), 
						  type = "{}".format(the_stream.output_datatype), 
						  channel_count = "{}".format(len(the_channels)), 
						  channel_format = "float32",
						  source_id = "{}".format(the_device))
		# __________________________________________________________________________________________
		info.desc().append_child_value("manufacturer", "Compmonks from Uchron Software")
		channels = info.desc().append_child("channels")
		for c in the_channels:
			channels.append_child("channel") \
					.append_child_value("label", c) \
					.append_child_value("unit", "microvolts") \
					.append_child_value("type", "{}".format(the_stream.output_datatype))
		self.outlet = StreamOutlet(info, self.the_chunk_num)
		xslog.stepLog("LSL stream output configured and ready: {} for the device {}".format(the_name,
																							the_device))
	#
	# METHODS_______________________________________________________________________________________
	def pushData(self, the_data):
		"""Push Data from a given pandas dataframe into a configured lsl outlet."""

		for ii in range(self.the_chunk_num):
			the_values = the_data.iloc[[ii]].values[:,1:] # the channel values
			the_timestamps = the_data.iloc[[ii]].values[:,0] # the timestamps values
			self.outlet.push_sample(the_data[:, ii], the_data)
#
####################################################################################################
if __name__ == '__main__':
	messageLog("ERROR")
	messageLog("This module should be imported instead of directly called")
####################################################################################################
#END		