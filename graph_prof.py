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
        self.backward_boundary_node = None
        self.forward_boundary_node = None
        self._analyze_graph()
        
        print(self.intermediate_nodes)
        print(len(self.intermediate_nodes))

        self.total_rumtime_sec : List[float] = []
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {} 
        self.compute_times = {}
        self.memory_usages = {}
    
        # for node in self.module.graph.nodes:
        #     print ("Node name: ", node.name)
        #     print ("Node type: ", node.op)
        #     print ("Node target: ", node.target)
        #     print ("Input to this node", node.all_input_nodes)
        #     print ("Users of this node: ", node.users)

    def _analyze_graph(self):
        seen_backward = False
        tensor_usage = {}
        # Iterate through nodes to find forward and backward boundary nodes
        for node in self.module.graph.nodes:
            if node.op == "call_function" and 'sep.default' in str(node.target):
                self.forward_boundary_node = node
            elif node.op == "call_function" and 'sep_backward.default' in str(node.target):
                self.backward_boundary_node = node
                seen_backward = True
                break
                # Initializing tracking for all nodes before backward boundary as potential intermediates
        
        if self.forward_boundary_node is not None and self.backward_boundary_node is not None:

            for node in self.module.graph.nodes:
                if node == self.backward_boundary_node:
                    break;
                if node.op not in [OP.PLACEHOLDER, OP.GET_ATTR, OP.OUTPUT]:
                    tensor_usage[node.name] = {
                        'first_fw_access': None,
                        'last_fw_access': None,
                        'first_bw_access': None,
                        'last_bw_access': None
                    }

            for node in self.module.graph.nodes:

                if node == self.forward_boundary_node:
                    break

                for user in node.users:
                    if user.name in tensor_usage:
                        if tensor_usage[user.name]['first_fw_access'] is None:
                            tensor_usage[user.name]['first_fw_access'] = node.name
                        tensor_usage[user.name]['last_fw_access'] = node.name
            if seen_backward:
                for node in reversed(list(tensor_usage)):
                    if node.op == OP.OUTPUT:
                        continue
                    for input_node in node.all_input_nodes:
                        if input_node.name in tensor_usage:
                            if tensor_usage[input_node.name]['last_bw_access'] is None:
                                tensor_usage[input_node.name]['last_bw_access'] = node.name
                            tensor_usage[input_node.name]['first_bw_access'] = node.name

                    if node == self.backward_boundary_node:
                        break

            # From tensor_usage get last_fw_uses and first_bw_uses

        
                    
            
    

    
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

        # self.env[n] = tensor
        
        
        # { node : (last_fw_uses){ (node -> tensor  )  } }

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
        self.compute_times[n.name] = start_event.elapsed_time(end_event)
        self.memory_usages[n.name] = end_memory - start_memory

        
        # you can end measuring the run-time of a node here
        # HINT: Use torch.cuda.Events for doing time measurements of operations.


        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.

        return result
