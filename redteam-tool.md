our methodology for red teaming agentic AIs (ais that use tools) will be as follows:

we will employ the existing HRL attacker (freeform attacker) with an additional layer of hierarchy.
Layer 1: top level policy is an MLP that ingests the base LLM's last layer hidden states and outputs an action in range (0,N), where N is the number of tools in the environment. the intuition is that the model reads the current state and predicts what tool will need to be used next. reward comes from the agentharm-gen reward functions (tools called in order, all tools called, etc)
Layer 2: the high level freeform policy; this reads the conversation and creates an instruction for the low level policy to follow, with the goal of getting the target agent to call the tool predicted by layer 1. layer 2 will determine what parameters and general strategy the low level agent should ask for. reward will be provided by the single tool-call instance, i.e. the target model called the tool specified by the upper level.
Layer 3: the low level freeform policy; this follows the above instructions and provides a prompt to the target model. the reward will be the token-level marginalization.

This methodology will be used to solve/train an agent for the agentharm-gen environment.