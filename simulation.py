import sys
import time
import concurrent.futures
import numpy as np
from functools import partial
from datasets import load_dataset
from datetime import datetime

from loguru import logger
from llm import LLM

sys_msg = 'You are a human user interacting with a conversational AI assistant or chatbot. You will engage in natural conversations, asking questions, making requests, and responding just as you would when talking to a real AI assistant. You can ask about any topics, request assistance with tasks, or just have an open-ended conversation. The key is to role-play as a typical user would when conversing with an AI chatbot, speaking in a casual, conversational manner and making your queries and responses sound like they would come from a human, not an AI itself. Do not break character or give any indication that you are an AI language model. Simply act as a realistic human user would when chatting with an AI assistant. Reply in no more than 500 words'

sys_prompt = {
    'role': 'system',
    'content': sys_msg
}
ds = load_dataset('HuggingFaceH4/ultrachat_200k')
ds = ds['train_gen']
dt = datetime.now()

logger.remove()
logger.add(f"./logs/{dt.strftime('%Y%m%d_%H%M')}.log", format="{message}", enqueue=True, level='INFO', serialize=True)


def user_llm() -> str:
    """
    User conversation is simulated as random messages
    from a dataset

    Args:
        None

    Returns
        {str} - A message sampled randomly from the dataset
    """
    idx = np.random.randint(0, len(ds))
    return ds[idx]['prompt']


def orchestrate_conversation(model: str, uri: str, conversation_id: int, num_total_turns: int) -> None:
    """
    Starts a chat between a simulated user and an LLM

    Args
        - model {str}:           Name of the model to call
        - uri {str}:                URI of the OpenAI compatible server
        - conversation_id {int}:    A unique identifier for the conversation
        - num_total_turns {int}:    Number of total turns to simulate the chat

    Returns
        None
    """
    logger.bind(conversation_id=conversation_id).info(f'Starting conversation with conversation id: {conversation_id}')

    chatbot_llm = LLM(
        model_id=model,
        uri=uri,
        messages=[]
    )
    user_msg = user_llm()

    conversation_turn = 0

    logger.bind(
        conversation_id=conversation_id
    ).info(f'Total number of turns for conversation id {conversation_id}\t{num_total_turns}')

    while conversation_turn < num_total_turns:
        try:
            logger.bind(
                conversation_id=conversation_id, turn=conversation_turn
            ).info(f'Conversation turn for conversation id {conversation_id}\t{conversation_turn}')
            chatbot_msg = chatbot_llm(message=user_msg, max_tokens=512)
            
            logger.bind(
                conversation_id=conversation_id, turn=conversation_turn
            ).info(f'Tokens: {chatbot_llm.total_tokens, chatbot_llm.input_tokens, chatbot_llm.output_tokens}')
            
            user_msg = user_llm()

        except Exception as err:
            logger.bind(
                conversation_id=conversation_id, turn=conversation_turn
            ).error(f'Error: {err} in conversation id {conversation_id}')
            logger.bind(
                conversation_id=conversation_id, turn=conversation_turn
            ).error(f'{chatbot_llm.messages}')
            break

        finally:
            conversation_turn += 1

    logger.bind(
        conversation_id=conversation_id
    ).info(f'Ending conversation {conversation_id} after {conversation_turn} turns')
    
    logger.bind(
        conversation_id=conversation_id
    ).info(f'Tokens: {chatbot_llm.total_tokens, chatbot_llm.input_tokens, chatbot_llm.output_tokens}')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=False, default='David-Xu/Mistral-7B-Instruct-v0.2', help='Model name that is called for inference')
    parser.add_argument('--uri', type=str, required=False, default='localhost', help='URI where the model is available for inference')
    parser.add_argument('--port', type=str, required=False, default='8000', help='Port where the model is available for inference')
    parser.add_argument('-r', type=float, required=False, default=None, help='Number of requests per second')
    parser.add_argument('-n', type=int, required=False, default=20, help='Number of requests to run')
    parser.add_argument('-l', type=int, required=False, default=15, help='Lower bound of conversations in a single chat')
    parser.add_argument('-u', type=int, required=False, default=25, help='Upper bound of conversations in a single chat')

    args = parser.parse_args()
    model = args.model
    complete_uri = f'http://{args.uri}:{args.port}/v1'
    rate = args.r
    time_unit = args.n
    l, u = args.l, args.u

    if rate:
        assert rate <= 1, f"Rate should be less than equal to 1. Provided: {rate}"

    logger.info(f'Model id: {model}, model uri: {complete_uri}')
    logger.info(f'Rate: {rate}, time units: {time_unit}')
    logger.info(f'Lower bound of total turns: {l}, higher bound of total turns: {u}')

    # Uniformly sample arrival times so that the mean is closer to the provided rate
    # This can also be replaced with sampling form poisson distribution to align better with the real world
    arrival_rates = np.zeros(time_unit)
    if rate:
        arrival_rates = np.random.randint(int(1/rate), int(1/rate)+1, time_unit)
    num_turns = np.random.randint(l, u, time_unit)

    logger.info(f'Arrvial rates: {arrival_rates}, num turns: {num_turns}')

    idx = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for arrival, num_total_turns in zip(arrival_rates, num_turns):
            logger.info(f'Sleeping for {arrival}s')
            if rate:
                time.sleep(arrival)

            f = partial(orchestrate_conversation, model=model, uri=complete_uri, conversation_id=idx, num_total_turns=num_total_turns)
            executor.submit(f)
            idx += 1

