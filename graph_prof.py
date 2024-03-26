from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation. 

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node
        # '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will 
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass. 

        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.

        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The number of parameters of the graphs and the gradients
        # should be equal.

        # You will also see several operators of the type 
        #' %tag_grad :[num_users=1] =
        # call_function[target=torch.ops.dummy.tag_grad.default]'. These are
        # also dummy operations added after every gradient produced in the
        # backward pass. 

        # Printing the input nodes, node users and node names.
        self.intermediate_nodes = {}
        self._analyze_graph()
        self.in_backward_pass = False
        
        # Fix:
        
        # for node in self.intermediate_nodes:
        #     print("Node name:", node)
        #     print(self.intermediate_nodes[node])
        # print(len(self.intermediate_nodes))

        self.compute_times = []
        self.memory_usages = []
        self.swap_time = []

        # for node in self.module.graph.nodes:

        
        # print ("Node name: ", self.backward_boundary_node.name)
        # print ("Node type: ", self.backward_boundary_node.op)
        # print ("Node target: ", self.backward_boundary_node.target)
        # print ("Input to this node", self.backward_boundary_node.all_input_nodes)
        # print ("Users of this node: ", self.backward_boundary_node.users)


    def _analyze_graph(self):
        seen_backward = False
        forward_nodes = set()
        node_usage = {}
        # Iterate through nodes to find forward and backward boundary nodes
        for node in self.module.graph.nodes:
            if node.op == "call_function" and node.name == 'sep':
                seen_backward = None
            elif node.op == "call_function" and node.name == 'sep_backward':
                seen_backward = True
                break

        for node in self.module.graph.nodes:
            if node.op == "call_function" and node.name == 'sep':
                break;
            forward_nodes.add(node);

        seen_backward = False
        for node in self.module.graph.nodes:

            if node.op == "call_function" and node.name == 'sep_backward':
                seen_backward = True

            if seen_backward:
                for input_node in node.all_input_nodes:
                    if input_node in forward_nodes and input_node.op != "placeholder":
                        node_usage[input_node.name] = {
                            'first_fw_access': None,
                            'last_fw_access': None,
                            'first_bw_access': None,
                            'last_bw_access': None
                        }
        
        seen_backward = False
        # for intermediate_node in node_usage.keys():
            
        for node in self.module.graph.nodes:
            if seen_backward == False:
                for input_node in node.all_input_nodes:
                    if input_node.name in node_usage:
                        if node_usage[input_node.name]['first_fw_access'] is None:
                            node_usage[input_node.name]['first_fw_access'] = node.name
                        node_usage[input_node.name]['last_fw_access'] = node.name
            if node.op == "call_function" and node.name == 'sep':
                seen_backward = None
            if node.op == "call_function" and node.name == 'sep_backward':
                seen_backward = True
            if seen_backward:
                for input_node in node.all_input_nodes:
                    if input_node.name in node_usage:
                        if node_usage[input_node.name]['first_bw_access'] is None:
                            node_usage[input_node.name]['first_bw_access'] = node.name
                        node_usage[input_node.name]['last_bw_access'] = node.name
                        
        # Fix
        
        # for node in node_usage:
        #     print("Node name: ", node)
        #     print(node_usage[node])
            

        # From tensor_usage get last_fw_uses and first_bw_uses

        for node in self.module.graph.nodes:
            fw_uses = set()
            bw_uses = set()
            for access in node_usage.keys():
                if node.name == node_usage[access]['last_fw_access']:
                    fw_uses.add(access)
                if node.name == node_usage[access]['first_bw_access']:
                    bw_uses.add(access)

                if len(fw_uses) != 0 or len(bw_uses) != 0:
                    self.intermediate_nodes[node.name] = {'last_fw_uses': fw_uses,
                                                'first_bw_uses': bw_uses}


    
    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> torch.Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        # self.env[n] : tensor
        if n.op == "call_function" and n.name == 'sep_backward':
            self.in_backward_pass = True

        if self.in_backward_pass:
            for input_name in n.all_input_nodes:
                if self.intermediate_nodes.get(input_name.name):
                   if n.name in self.intermediate_nodes[input_name.name]['first_bw_uses']:
                       # swap in/out
                       swap_start = torch.cuda.Event(enable_timing=True)
                       swap_end = torch.cuda.Event(enable_timing=True)
                       swap_start.record()
                       
                       temp = self.env.get(n)
                       if torch.is_tensor(temp):
                           self.env[n] = temp.device(torch.device('cuda:0'))
                           del temp
                       swap_end.record()
                       torch.cuda.synchronize()
                       self.swap_time.append({"node": n.name, "transfer": "CPU -> GPU", "time" : swap_start.elapsed_time(swap_end)})

        # you can start measuring the run-time of a node here
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_memory = torch.cuda.memory_allocated()

        # Run the node
        result = super().run_node(n)

        # End timing and memory tracking
        end_event.record()
        torch.cuda.synchronize()  # Ensure timing is accurate
        end_memory = torch.cuda.memory_allocated()

        # Record compute time and memory usage
        self.compute_times.append({"node": n.name, "time" : start_event.elapsed_time(end_event)})
        self.memory_usages.append({"node": n.name, "time" : end_memory})

        
        # you can end measuring the run-time of a node here
        # HINT: Use torch.cuda.Events for doing time measurements of operations.


        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        
        if not self.in_backward_pass:
            for user in n.users:
                if self.intermediate_nodes.get(user.name):
                    if n.name in self.intermediate_nodes[user.name]['last_fw_uses']:
                        
                        swap_start = torch.cuda.Event(enable_timing=True)
                        swap_end = torch.cuda.Event(enable_timing=True)
                        swap_start.record()
                        
                        temp = self.env.get(n)
                        if torch.is_tensor(temp):
                            self.env[n] = temp.device(torch.device('cpu'))
                            del temp

                        swap_end.record()
                        torch.cuda.synchronize()
                        self.swap_time.append({"node": n.name, "transfer": "GPU -> CPU", "time": swap_start.elapsed_time(swap_end)})
        
        
         
        return result
