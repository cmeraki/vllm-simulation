# vllm-simulation

This is the accompanying repo to the blog [Throughput is all you need](https://cmeraki.github.io/blogs/throughput.html). The code here simulates a chat application, where a user engages with a LLM powered bot in a multi turn conversation.

## Setup

Step 0: Setup the environment

```bash
$ git clone https://github.com/cmeraki/vllm-simulation.git
$ cd vllm-simulation
$ python -m venv myenv
$ source myenv/bin/activate
```

Step 1: Setup vLLM following the guide [here]().

Step 2: Run the simulation

To replicate the experiments from the blog, we would need three running processes.

1. vLLM serving an LLM on an OpenAI compatible server. After installing the setup, run this command in a terminal window:

```bash
$ python3 <>
```

2. Monitoring setup: To set up the monitoring, follow the steps mentioned [here](https://github.com/vllm-project/vllm/tree/main/examples/production_monitoring). This will be important to visualize the metrics.
3. Simulation: To finally run the simulation, open a terminal window in this repository and run the following command

```bash
python --model_id togethercomputer/Llama-2-7B-32K-Instruct \
	-r 0.1 -n 100 -l 9 -u 11 \
	simulation.py
```

You can customize the following arguments to the script.

```bash
python simulation.py -h
usage: simulation.py [-h] [--model_id MODEL_ID] [--uri URI] [--port PORT] [-r R] [-n N] [-l L] [-u U]

options:
  -h, --help           show this help message and exit
  --model_id MODEL_ID  Model name that is called for inference
  --uri URI            URI where the model is available for inference
  --port PORT          Port where the model is available for inference
  -r R                 Number of requests per second
  -n N                 Time unit to run the test for
  -l L                 Lower bound of conversations in a single chat
  -u U                 Upper bound of conversations in a single chat
```

