from .claude import *
from .utils import *
from typing import Tuple
from copy import deepcopy
import json


MAX_ITER = 1
PRINT_FUNC_CALL = True
PRINT_FUNC_RESULT = True
set_log_enable(True)


# pseudo code:

# MAX_ITER = 5
# messages = []
# model = <the_ai_assistant>  # This is you
#
# def user_input(question: str) -> str:
#     function_call_num = 0
#     messages.append({"role": "user", "content": question})
#     while True:
#         output = model.get_response(messages)
#         if (output.is_to_user):
#             messages.append({"role": "assistant", "content": output})
#             return output.text
#         else:
#             # to system
#             if (function_call_num == MAX_ITER):
#                 assert (False)  # This should not happen, because you should not do function callings when it reaches MAX_ITER
#             function_call_num += 1
#             messages.append({"role": "assistant", "content": output})
#             functions_list = output.functions
#             function_results = run_functions(functions_list)
#             messages.append({"role": "user", "content": function_results})


def chat_one_round_glkb(messages_history: list[dict], question: str) -> Tuple[list[dict], str]:
    '''
    return (messages_history, response)
    '''
    question = question.strip()
    if (question == ''):
        question = '<empty>'
    question = '====== From User ======\n' + question
    messages = deepcopy(messages_history)
    messages.append({"role": "user", "content": question})
    function_call_num = 0
    while True:
        messages, response = chat_and_get_formatted(messages)
        if (response['to'] == 'user'):
            return (messages, response['text'])
        if (function_call_num == MAX_ITER):
            assert (False)  # Currently not handle this error
        function_call_num += 1
        functions_result = run_functions(response['functions'])
        
        # OPTIMIZATION: Return raw HIRN results immediately instead of another GPT synthesis call
        # This eliminates one redundant GPT call - FormatAgent will handle synthesis
        raw_response = {
            'source': 'glkb',
            'query_used': response.get('functions', []),
            'raw_abstracts': functions_result
        }
        return (messages, json.dumps(raw_response))


def chat_forever():
    messages = []
    while True:
        question = input('Your question: ')
        messages, response = chat_one_round_glkb(messages, question)
        print(f'\nResponse:\n\n{response}\n')


if __name__ == "__main__":
    chat_forever()
