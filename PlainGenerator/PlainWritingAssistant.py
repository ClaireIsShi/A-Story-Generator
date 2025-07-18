'''
-- @Time    : 2025/7/10 16:15
-- @File    : PlainWritingAssistant.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

from settings import UTIL_LLM

## Create a plain story generator assistant
#Invocation method:
"""
writing_assistant = PlainWritingAssistant(
                language="English",
                length=350,
                topic="Adventure",
                last_outline="The hero set out on a journey to find the lost treasure.",
                goal="Find the treasure and become rich",
                long_term_memory="The hero heard rumors about the treasure from an old sage."
            )
chosen_outline = writing_assistant()
"""
import os,sys,json
from langchain_openai import ChatOpenAI
from typing import Optional
import os,sys

import warnings

warnings.filterwarnings("ignore")
from utils import get_content_between_a_b,set_env
set_env()

# Add the parent directory to sys.path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

GENERATE_PROMPT="""
You're a story generator and a native speaker of {language}. Your task is to generate outlines in {language} continuing write this story based on {last_outline}, your outline must be close to topic: '{topic}' and the goal: '{goal}'. Bsed on your memory:{long_term_memory}
Follow these steps:
1. create 3 DIFFERENT continuing outlines based on the original outline at least at {length} words;
2. Your output should be in {language}, and your story should be still across to this topic: {topic}.
Output your result in the following format, don't change the format English, such as "## Outline1:":
## Outline1:
<here put your first outline in {language}>
## Outline2:
<here put your second different outline in {language}>
## Outline3:
<here put your third different outline in {language}>
## END
"""



###########################
SELECT_PROMPT="""
selecting the most funny continuing outlines from the following three outlines:
{outline1},{outline2},{outline3}, and give me your reason. 
Follow these steps:
1. analysis these three outlines;
2. select the one most funny continuing outlines from these three outlines;
Output your result in the following format,don't change the format English, such as"## Reason:":
## Reason:
<here put your reason>
## Selected Outline:
< here put your selected outline of these 3 choices>
## END
"""


def parser_generate_response(response:str):
    """
    Function to parse the response of generated story outlines.

    This function extracts three different story outlines from the response text returned by the large language model.
    It uses the `get_content_between_a_b` function to intercept the corresponding outline content based on specific delimiters.

    Args:
        response (str): The response text containing story outlines returned by the large language model.

    Returns:
        tuple: A tuple containing three story outline strings, namely the first outline, the second outline, and the third outline.
    """
    # Extract the first story outline from the response text
    outline1 = get_content_between_a_b("## Outline1:","## Outline2:",response)
    # Extract the second story outline from the response text
    outline2 = get_content_between_a_b("## Outline2:","## Outline3:",response)
    # Extract the third story outline from the response text
    outline3 = get_content_between_a_b("## Outline3:","## END",response)
    return outline1,outline2,outline3


def parser_select_response(response:str):
    """
    Function to parse the response of selected story outlines.

    This function extracts the reason for selecting the story outline and the final selected story outline from the response text returned by the large language model.
    It uses the `get_content_between_a_b` function to intercept the corresponding content based on specific delimiters.

    Args:
        response (str): The response text containing the selection reason and the selected outline returned by the large language model.

    Returns:
        tuple: A tuple containing the selection reason string and the selected story outline string.
    """
    # Extract the reason for selecting the story outline from the response text
    reason = get_content_between_a_b("## Reason:","## Selected Outline:",response)
    # Extract the final selected story outline from the response text
    selected = get_content_between_a_b("## Selected Outline:","## END",response)
    return reason,selected


class PlainWritingAssistant:
    def __init__(self, language="English", length=350, topic="",last_outline="", goal = "",long_term_memory:str = '',start_sign:bool=False, llm=UTIL_LLM):
        """
        Initialize an instance of the PlainWritingAssistant class.

        Args:
            language (str, optional): The language used for generating and selecting story outlines, defaults to "English".
            length (int, optional): The minimum word count for the generated story outline, defaults to 350.
            topic (str, optional): The theme of the story, defaults to an empty string.
            last_outline (str, optional): The previous story outline, defaults to an empty string.
            goal (str, optional): The goal of the story, defaults to an empty string.
            long_term_memory (str, optional): Long-term memory content, defaults to an empty string.
            llm (ChatOpenAI, optional): The instance used to call the large language model, defaults to a ChatOpenAI instance using the gpt-3.5-turbo model.
        """
        self.start_sign = start_sign
        self.language = language
        self.length = length
        self.topic = topic
        self.goal = goal
        self.last_outline = last_outline
        self.llm = llm
        self.long_term_memory = long_term_memory
        print("Setting up PlainWritingAssistant...")

    def __call__(self, storage:Optional[str] = None)->str:
        """
        Method executed when the class instance is called.
        This method calls the `step` method and returns the return value of the `step` method.

        Returns:
            The return value of calling the `step` method.
        """
        button = True
        trying = 0
        while button and trying < 4:
            try:
                self.clear()
                button = False
                return self.step(storage)
            except:
                print(f"running {self.__class__.__name__} failed, retrying...")
                trying += 1
        warnings.warn(f"running {self.__class__.__name__} failed {trying} times, return None")
        return None



    def generate(self):
        """
        Generate a prompt based on a predefined template and call the large language model to get the response content.

        Returns:
            str: The response text returned by the large language model.
        """
        prompt = GENERATE_PROMPT.format(language=self.language, length=self.length, topic=self.topic, goal=self.goal, last_outline=self.last_outline, long_term_memory=self.long_term_memory)
        response = self.llm.invoke(prompt).content
        return response

    def generate_outlines(self):
        """
        Call the generate method to get the response from the large language model, and parse the response to get three different story outlines.

        Returns:
            tuple: A tuple containing three story outline strings, namely the first outline, the second outline, and the third outline.
        """
        button = True
        trying = 0

        response = self.generate()
        self.outline1,self.outline2,self.outline3 = parser_generate_response(response)

        return self.outline1,self.outline2,self.outline3

    def select_outlines(self, show_reason=False):
        """
        Generate three story outlines and select the most interesting one from them.

        Args:
            show_reason (bool, optional): Whether to return the reason for selection, defaults to False.

        Returns:
            Union[str, tuple]: If show_reason is False, return the selected story outline; if True, return a tuple containing the selection reason and the selected story outline.
        """
        self.generate_outlines()
        prompt = SELECT_PROMPT.format(length=self.length, topic=self.topic, goal=self.goal, outline1=self.outline1, outline2=self.outline2, outline3=self.outline3, start_sign=self.start_sign)
        self._reason, self.chosen_outline = parser_select_response(self.llm.invoke(prompt).content)
        if show_reason:
            return self._reason,self.chosen_outline
        else:
            return self.chosen_outline

    def step(self, mem_storage: Optional[str] = None):
        """
        Execute the steps of generating and selecting story outlines, and store the results if needed.

        Args:
            mem_storage (Optional[str], optional): The file path to store the summary and the selected outline, defaults to None.

        Returns:
            str: The selected story outline.
        """
        if mem_storage:
            import json
            self.select_outlines(show_reason=True)
            summarization = self.llm.invoke("here is an outline:{outline}, summerize it in up to 30 words.".format(outline = self.chosen_outline)).content
            k_v = {
                summarization.split("\n"): self.chosen_outline
            }
            with open(mem_storage, "w", encoding = 'UTF-8') as f:
                json.dump(k_v, f,ensure_ascii=False, indent=2)
        else:
            self.chosen_outline = self.select_outlines()
        return self.chosen_outline

    def clear(self):
        """
        Reset the attributes generated during the process of generating and selecting story outlines to None.

        Returns:
            PlainWritingAssistant: The current instance of the class.
        """
        # Define a list of attributes to be reset to None
        attrs_to_clear = ['outline1' , 'outline2' , 'outline3' , 'chosen_outline' , '_reason']
        # Iterate through the attribute list and use the setattr function to set the attribute values to None
        if hasattr(self, 'outline1'):
            for attr in attrs_to_clear:
                delattr(self, attr)
        return self