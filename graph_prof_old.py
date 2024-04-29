from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any

import json
import gc


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

        # for node in self.module.graph.nodes:
        #     print("Node name: ", node.name)
        #     print("Node type: ", node.op)
        #     print("Node target: ", node.target)
        #     print("Input to this node", node.all_input_nodes)
        #     print("Users of this node: ", node.users)
        #     print()

        # Each key is a node; each value indicates whether that node's output is a
        # 'parameter', 'activation', 'gradient', 'optimzer_state', or 'other'
        self.node_types = {}
        # Nodes that are candidates for being activations:
        # created in forward pass and are not parameters, but not have yet
        # confirmed whether the nodes appear in backward pass
        activation_candidates = set()
        # Nodes that are confirmed to be activations
        activations = set()
        # Keeps track of state in the following loop,
        # indicating 'forward' pass, 'backward' pass, or 'neither'
        cur_pass = 'forward'
        # Loop through all nodes in order
        for node in self.module.graph.nodes:
            # 'sep' node indicates end of forward pass
            if node.name == 'sep':
                cur_pass = 'neither'
                self.node_types[node] = 'other'
            # 'sep_backward' node indicates start of backward pass
            elif node.name == 'sep_backward':
                cur_pass = 'backward'
                self.node_types[node] = 'other'
            # Node is between forward and backward passes
            elif cur_pass == 'neither':
                self.node_types[node] = 'other'
            # Forward pass
            elif cur_pass == 'forward':
                # Node has no inputs -> node is a parameter
                if not node.all_input_nodes:
                    self.node_types[node] = 'parameter'
                # Not a parameter -> could be an activation
                else:
                    activation_candidates.add(node)
            # Backward pass
            elif cur_pass == 'backward':
                # Node is tagged as gradient; i.e., the next node is the dummy gradient tag
                if node.next.target == torch.ops.dummy.tag_grad.default:
                    self.node_types[node] = 'gradient'
                # TODO: handle 'optimizer_state'
                else:
                    self.node_types[node] = 'other'
                # If some node `n` used by this backward-pass node is an activation candidate,
                # we conclude `n` is indeed an activation
                for n in node.all_input_nodes:
                    if n in activation_candidates:
                        self.node_types[n] = 'activation'
                        activation_candidates.remove(n)
                        activations.add(n)
        # Activation candidates that turned out not be activations are classified as 'other'
        for n in activation_candidates:
            self.node_types[n] = 'other'

        # Each key is an activation node, each value is a 2-item list containing
        # [last node in forward pass that uses the key,
        #  first node in backward pass that uses the key]
        self.activation_unused_range = {}
        # True during forward pass of the following loop, False during backward pass
        is_forward_pass = True
        # Loop through all nodes in order
        for node in self.module.graph.nodes:
            if node.name == 'sep_backward':
                is_forward_pass = False
            for n in node.all_input_nodes:
                # Identify all activation nodes `n` used by `node`
                if n in activations:
                    # In forward pass, overwrite as much as needed so that
                    # the last occurrence is recorded
                    if is_forward_pass:
                        self.activation_unused_range[n] = [node, None]
                    # In backward pass, write only once so that
                    # the first occurrence is recorded
                    elif self.activation_unused_range[n][1] is None:
                        self.activation_unused_range[n][1] = node
            # In forward pass, last occurrence of activation may be the moment it is created
            if is_forward_pass and node in activations:
                self.activation_unused_range[node] = [node, None]
        
        # `forward_last_users[k] = [v1, v2, ...]` indicates that node `k` is
        # the last node in the forward pass to use the activation nodes `v1`, `v2`, ...
        self.forward_last_users = {}
        # `backward_first_users[k] = [v1, v2, ...]` indicates that node `k` is
        # the first node in the backward pass to use the activation nodes `v1`, `v2`, ...
        self.backward_first_users = {}
        for activation_node, (forward_last, backward_first) in self.activation_unused_range.items():
            if forward_last not in self.forward_last_users:
                self.forward_last_users[forward_last] = []
            if backward_first not in self.backward_first_users:
                self.backward_first_users[backward_first] = []
            self.forward_last_users[forward_last].append(activation_node)
            self.backward_first_users[backward_first].append(activation_node)

        # Save information for use in analysis
        self.stats = {}
        
        # True during forward pass, False during backward pass
        # Keeps track of state inside `run_node()`
        self.is_forward_pass = True

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> torch.Any:
        ret = super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )
        with open('stats.json', 'w') as f:
            json.dump(self.stats, f)
        peak_mem = max(v['memory'] for v in self.stats.values())
        print(peak_mem)
        return ret

    def run_node(self, n: fx.Node) -> Any:

        # End of forward pass
        if n.name == 'sep':
            self.is_forward_pass = False

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.
        if not self.is_forward_pass and n in self.backward_first_users:
            for x in self.backward_first_users[n]:
                self.env[x] = self.env[x].cuda()

        # Prepare to measure GPU memory usage
        torch.cuda.reset_peak_memory_stats(device=None)
        
        # you can start measuring the run-time of a node here
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        result = super().run_node(n)

        # you can end measuring the run-time of a node here
        # HINT: Use torch.cuda.Events for doing time measurements of operations.
        end_time.record()
        torch.cuda.synchronize()
        
        # Info for analysis
        self.stats[n.name] = {
            'type': self.node_types[n],
            'time': start_time.elapsed_time(end_time),
            'memory': torch.cuda.max_memory_allocated()
        }
        if self.node_types[n] == 'activation':
            activation_unused_range = [node.name for node in self.activation_unused_range[n]]
            self.stats[n.name]['activation_unused_range'] = activation_unused_range

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        if self.is_forward_pass and n in self.forward_last_users:
            for x in self.forward_last_users[n]:
                if x == n:
                    # `x` is the node currently being run
                    result = result.cpu()
                else:
                    # `x` is a previously-run node
                    self.env[x] = self.env[x].cpu()
        
        return result
