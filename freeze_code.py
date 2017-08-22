from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#from TORCS_params import parse_args
from tensorflow.python.tools import freeze_graph


# saved trained-net
net_dir = '/home/daehwakim/freeze'
checkpoint_prefix = os.path.join(net_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "graph.pb"

input_graph_path = os.path.join(net_dir, input_graph_name)
input_saver_def_path = ""
input_binary = True
input_checkpoint_path = os.path.join(net_dir, 'saved_checkpoint') + "-0"


# Note that we this normally should be only "output_node"!!!
output_node_names = "hypothesis"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join(net_dir, output_graph_name)
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, input_checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_graph_path, clear_devices, "")
