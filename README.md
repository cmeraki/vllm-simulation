# vLLM-simulation

This is the accompanying repo to the blog [Throughput is all you need](https://cmeraki.github.io/blogs/throughput.html). The code here simulates a chat application, where a user engages with an LLM powered bot in a multi-turn conversation.

Here, we have used a flavor of Mistral 7B model developed by Teknium. You can find more information about them [here](https://huggingface.co/teknium).

## Setup

### Prerequisites

You need to have the following tools setup in your system

1. git
2. Docker CLI and Docker compose
3. Python 3

### Step 0: Setup the environment

```bash
git clone https://github.com/cmeraki/vllm-simulation.git
cd vllm-simulation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Run the simulation

To replicate the experiments from the blog, we would need three running processes.

1. vLLM serving an LLM on an OpenAI-compatible server. After running step 0, run this command in a new terminal window:

```bash
python -m vllm.entrypoints.openai.api_server --model teknium/OpenHermes-2.5-Mistral-7B --max-model-len 8192 --disable-log-requests
```

2. Monitoring setup: To set up the monitoring, follow the steps mentioned [here](https://github.com/vllm-project/vllm/tree/main/examples/production_monitoring). This will be important to visualize the metrics. To run the code in the given link above, you can either download the code or clone the repository itself.

3. Simulation: To finally run the simulation, open a new terminal window in this repository location and run the following command

```bash
python simulation.py --model teknium/OpenHermes-2.5-Mistral-7B -n 50 -l 10 -u 11
```

You can customize the following arguments to the above script.

```bash
python simulation.py -h
usage: simulation.py [-h] [--model MODEL] [--uri URI] [--port PORT] [-r R] [-n N] [-l L] [-u U]

options:
  -h, --help           show this help message and exit
  --model MODEL        Model name that is called for inference
  --uri URI            URI where the model is available for inference
  --port PORT          Port where the model is available for inference
  -r R                 Number of requests per second
  -n N                 Time unit to run the test for
  -l L                 Lower bound of conversations in a single chat
  -u U                 Upper bound of conversations in a single chat
```

Given that you have run all the steps successfully, you would be able to visualize the metrics at http://localhost.com:3000.

## Appendix

The simulation was run on an Nvidia RTX 4090 GPU.
