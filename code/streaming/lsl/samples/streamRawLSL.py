#streamRawLSL.py
#
name             = "streamRawLSL"
version          = "0.0.1+git" #
author           = "Pierre Cutellic"
author_email     = "cutellic@arch.ethz.ch"
home_page        = "http://www.compmonks.com/"
license          = "License" # License Type
description      = "Stream data from an OpenBCI device into a thread with LSL format."

long_description = \
"""
Yapsy plugin to stream an OpenBCI device stream into a separate thread with LSL format.
Typically stream types can be EEG, AUX (accelerometer) and IMP (impedance) data.

"""
#
####################################################################################################
# TODO + WARNINGS
# - ...
####################################################################################################
# MODULES
import sys,os
#sys.path.append('./neurodesign_app/static/neurodesign_app/python') # include obci custom packages
#sys.path.append('./neurodesign_app/static/neurodesign_app')
#sys.path.append('./neurodesign_app/static/neurodesign_app/python/plugins/bin') # lsl libraries
sys.path.append(os.path.join(".","neurodesign_app","static","neurodesign_app","python"))
sys.path.append(os.path.join(".","neurodesign_app","static","neurodesign_app"))
sys.path.append(os.path.join(".","neurodesign_app","static","neurodesign_app","python","plugins","bin")) # include custom packages
#sys.path.append(os.path.join("..",""))
#
try:
	from pylsl import StreamInfo, StreamOutlet
	from utilities.logging import xslog
	from utilities.devices.openbci import plugin_interface as plugintypes
	import time

except():
	print(sys.exc_info())
	sys.exit(1)
#	
# VERSION CHECKS
# print('Python: {}'.format(sys.version))
# print('plugintypes: {}'.format(plugintypes.__version__))
# print('dataPackages: {}'.format(dataPackages.__version__))
#
####################################################################################################
# VARS
#
####################################################################################################
# RESSOURCES
# ressources preloading classes or methods
#
# CLASSES __________________________________________________________________________________________
# 
class streamRawLSL(plugintypes.IPluginExtended):
	"""Use LSL protocol to broadcast data using one stream for EEG, one stream for AUX, 
		one last for impedance testing (on supported board, if enabled)"""

	#
	# CONSTRUCTOR___________________________________________________________________________________
	def __init__(self, args = []):
		"""..."""

		self.the_prefix = "OpenBCI"
		self.args = args
		self.labels_eeg = []
		self.labels_aux = []
		self.labels_imp = []
		self.outlet_eeg = None
		self.outlet_aux = None
		self.outlet_imp = None
		self.chunk_size = 12 # 1024
		self.max_buffered = 360

	# ______________________________________________________________________________________________
	def activate(self):
		"""plugin activation."""

		time.sleep(.5)		
		if len(self.args) > 0:
			# Create a new streams info, one for EEG values, one for AUX (eg, accelerometer) values
			# set float32 instead of int16 so as OpenViBE takes it into account
			for arg_ in self.args:
				if "{}".format(type(arg_)) == "<class 'dict'>":
					xslog.infoLog("Getting arguments for channels")
					xslog.infoLog("PLUGIN ARGS: {}".format(self.args))
					if 'EEG' in arg_:
						self.labels_eeg = arg_["EEG"]
					else:
						self.labels_eeg = []
					if 'ACC' in arg_:
						self.labels_aux = arg_["ACC"]
					else:
						self.labels_aux = []
					if 'IMP' in arg_:
						self.labels_imp = arg_["IMP"]
					else:
						self.labels_imp = []
					if 'Device' in arg_:
						self.the_prefix = arg_["Device"] 
					break

				else:
					xslog.infoLog("No channel has been passed to the plugin. Check the driver.")
					xslog.infoLog("PLUGIN ARGS: {}".format(self.args))
					xslog.infoLog("types: {}".format([type(arg_) for arg_ in self.args]))
			#
			try:
				if len(self.labels_eeg + self.labels_aux + self.labels_imp) > 0:
					#
					if len(self.labels_eeg) > 0:
						eeg_name = "{} LSL EEG RAW".format(self.the_prefix)
						xslog.stepLog("Creating LSL stream for EEG. Name: {}".format(eeg_name))
						xslog.infoLog("source_id: {}".format(self.the_prefix))
						xslog.infoLog("data type: float32.")
						xslog.infoLog("{} channel(s) at {} Hz.".format(len(self.labels_eeg),self.sample_rate))
						info_eeg = StreamInfo(eeg_name, 
											  'EEG', 
											  len(self.labels_eeg),
											  self.sample_rate,
											  'float32',
											  self.the_prefix)
						info_eeg.desc().append_child_value("manufacturer", "OpenBCI")
						channels_eeg = info_eeg.desc().append_child("channels")
						for c in self.labels_eeg[:self.eeg_channels]:
							channels_eeg.append_child("channel")\
										.append_child_value("name", c)\
										.append_child_value("unit", "microvolts")\
										.append_child_value("type", "EEG")
						#self.outlet_eeg = StreamOutlet(info_eeg,self.chunk_size,self.max_buffered)
						self.outlet_eeg = StreamOutlet(info_eeg)
						#
					if len(self.labels_aux) > 0:
						aux_name = "{} LSL ACC RAW".format(self.the_prefix)
						xslog.stepLog("Creating LSL stream for AUX. Name: {}".format(aux_name))
						xslog.infoLog("source_id: {}".format(self.the_prefix))
						xslog.infoLog("data type: float32.")
						xslog.infoLog("{} channel(s) at {} Hz.".format(len(self.labels_aux),self.sample_rate))
						info_aux = StreamInfo(aux_name,
											  'ACC', 
											  len(self.labels_aux),
											  self.sample_rate,
											  'float32',
											  self.the_prefix)
						info_aux.desc().append_child_value("manufacturer", "OpenBCI")
						channels_aux = info_aux.desc().append_child("channels")
						for c in self.labels_aux[:self.aux_channels]:
				 			channels_aux.append_child("channel")\
										.append_child_value("name", c)\
										.append_child_value("unit", "acc G")\
										.append_child_value("type", "ACC")
						#self.outlet_aux = StreamOutlet(info_aux,self.chunk_size,self.max_buffered)
						self.outlet_aux = StreamOutlet(info_aux)
						#
					if len(self.labels_imp) > 0 and self.imp_channels > 0:
						imp_name = "{} LSL IMP RAW".format(self.the_prefix)
						xslog.stepLog("Creating LSL stream for Impedance. Name: {}".format(imp_name))
						xslog.infoLog("source_id: {}".format(self.the_prefix))
						xslog.infoLog("data type: float32.")
						xslog.infoLog("{} channel(s) at {} Hz.".format(len(self.labels_imp),self.sample_rate))
						info_imp = StreamInfo(imp_name,
											  'IMP', 
											  len(self.labels_imp),
											  self.sample_rate,
											  'float32',
											  self.the_prefix)
						info_imp.desc().append_child_value("manufacturer", "OpenBCI")
						channels_imp = info_imp.desc().append_child("channels")
						for c in self.labels_imp[:self.imp_channels]:
							channels_imp.append_child("channel")\
										.append_child_value("name", c)\
										.append_child_value("unit", "microvolts")\
										.append_child_value("type", "IMP")
						#self.outlet_imp = StreamOutlet(info_imp,self.chunk_size,self.max_buffered)
						self.outlet_imp = StreamOutlet(info_imp)
			except Exception as e:
				xslog.infoLog("OPENBCI PLUGIN STREAM RAW LSL ERROR: {}".format(e))
				self.deactivate()	
	# ______________________________________________________________________________________________
	def __call__(self, sample):
		"""send channels values."""

		if sample:
			if self.outlet_eeg is not None:
				self.outlet_eeg.push_sample(sample.channel_data,time.time())
				#xslog.infoLog(sample.channel_data)
			if self.outlet_aux is not None:
				#if sample.aux_data.count(0.0) != len(sample.aux_data): # try to not send null samples with all 0.0 values
				self.outlet_aux.push_sample(sample.aux_data,time.time())
				# xslog.infoLog(sample.aux_data)
			if self.outlet_imp is not None:
				self.outlet_imp.push_sample(sample.imp_data,time.time())
	# ______________________________________________________________________________________________
	def deactivate(self):
		"""plugin deactivation"""

		xslog.infoLog("streamRawLSL plugin deactivated")
		return
	# ______________________________________________________________________________________________
	def show_help(self):
		"""..."""

		xslog.stepLog("Stream an OpenBCI stream into a thread with LSL format.")
#
####################################################################################################
if __name__ == '__main__':
	xslog.stepLog("ERROR")
	xslog.infoLog("This plugin is not meant to be called directly.\
				   Import it to use it in your code instead.")
####################################################################################################
#END