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
        
        for node in self.module.graph.nodes:
            if node == self.backward_boundary_node:
                break
            if node.op not in [OP.PLACEHOLDER, OP.GET_ATTR, OP.OUTPUT]:
                tensor_usage[node] = {
                    'last_fw_use': None,
                    'first_bw_use': None,
                }

        # Track usage in the forward pass
        for node in self.module.graph.nodes:
            if node == self.forward_boundary_node:
                break
            for user in node.users:
                if user in tensor_usage:
                    tensor_usage[user]['last_fw_use'] = node

        # Track usage in the backward pass, starting from the backward boundary
        if seen_backward:
            for node in list(reversed(list(self.module.graph.nodes))):
                if node.op == OP.OUTPUT:
                    continue
                for input_node in node.all_input_nodes:
                    if input_node in tensor_usage and tensor_usage[input_node]['first_bw_use'] is None:
                        tensor_usage[input_node]['first_bw_use'] = node
                if node == self.backward_boundary_node:
                    break

        for node in tensor_usage:

            if tensor_usage[node]['last_fw_use'] is not None and tensor_usage[node]['first_bw_use'] is not None:

                self.intermediate_nodes[node.name] = {
                    'last_fw_uses': tensor_usage[node]['last_fw_use'],
                    'fist_bw_uses': tensor_usage[node]['first_bw_use']
                }
    

    
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

    def _handle_forward_node(self, n: fx.Node):
    # Check if any input tensor is last used here and swap it out
        for input_name in n.all_input_names:
            if input_name in self.last_uses and self.last_uses[input_name] == n.name:
                tensor = self.fetch_attr(input_name)
                if tensor.is_cuda:
                    # Move the tensor to CPU and record this action
                    self.swapped_tensors[input_name] = tensor.cpu()
                    print(f"Swapped out tensor '{input_name}' to CPU")

    def _handle_backward_node(self, n: fx.Node):
        # Check if this node uses any tensor that was swapped out
        for input_name in n.all_input_names:
            if input_name in self.swapped_tensors:
                # Move the tensor back to GPU
                tensor = self.swapped_tensors[input_name].cuda()
                self.swapped_tensors.pop(input_name)  # Remove from swapped list
                print(f"Swapped in tensor '{input_name}' to GPU")
