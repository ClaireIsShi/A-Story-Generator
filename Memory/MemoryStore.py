'''
-- @Time    : 2025/7/14 23:47
-- @File    : MemoryStore.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

import os, json,sys

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import get_content_between_a_b,set_env


from utils import set_env
from StoryState import StoryState
from settings import STORY_SETTING_PATH, MEMORY_STORAGE_PATH
set_env()
import warnings
SYS_MEMORY_PROMPT = """
You're a good storage bot for saving story outlines.You're a native {language} speaker. You're good at summary stories and save them in logical order. You got a story outline summarization job, the settings of the story are as follows:
topic: {topic}, Main character: {main_character}, Main Goal:{main_goal} language: {language}.
"""
# need to enter:{
#     'topic': topic,
#     'main_character': main_character,
#     'main_goal': main_goal,
#    'language': language,
#    'first_outline':first_outline,
# }
BEGINNING_SYS_MEMORY_PROMPT = """
Here's the first outline of this story:{first_outline}
save it properly in your memory. Save it as short as possible, but make sure important parts are saved.
return your memory storage as following format:
## new memory added:
<here put memory you think is important in this first outline>
## END
"""

WRITE_MEMORY_PROMPT = """
Here's a new outline of the story:{new_outline}. Save it clearly and properly in your memory, based on the memories you already saved:{memory_storage}.Save it as short as possible, but make sure important parts are saved.
return your memory storage as following format:
## new memory added:
<here put memory you think is important in your new outline>
## END
"""

def memory_parser(str)->str:
    """
    Parses the response string to extract the content between '## new memory added:' and '## END'.
    :param str: The input string to be parsed.
    :return: The extracted content as a string.
    """
    return get_content_between_a_b('## new memory added:','## END',str)

from settings import UTIL_LLM
llm = UTIL_LLM


class MemoryStore:
    def __init__(self, state:StoryState, llm = llm):
        """
        Initializes the MemoryStore instance.
        :param state: The StoryState object containing story metadata and recent story content.
        :param llm: The language model instance (default is ChatOpenAI with gpt-3.5-turbo).
        """
        self.state = state
        self.llm = llm
        self.memory_store = None

    def __call__(self):
        return self.memory_store

    def first_store(self):
        """
        Stores the first outline of the story into memory using the language model.
        Initializes the memory_store attribute with the parsed response.
        """
        try:
            print(f"Generating memory...")
            system_message_prompt = SystemMessagePromptTemplate.from_template ( SYS_MEMORY_PROMPT )
            human_message = HumanMessagePromptTemplate.from_template ( BEGINNING_SYS_MEMORY_PROMPT+BEGINNING_SYS_MEMORY_PROMPT )
            prompt_setting = ChatPromptTemplate.from_messages ( [system_message_prompt , human_message] )
            init_prompt = prompt_setting.format (
                topic=self.state['Topic'] ,
                main_character=self.state['MainCharacter'] ,
                main_goal=self.state['MainGoal'] ,
                language=self.state['Language'] ,
                first_outline=self.state["RecentStory"][0]
            )
            response = self.llm.invoke ( [init_prompt] ).content
            memory = memory_parser(response)
            self.memory_store = memory
        except:
            warnings.warn(f"Memory store could not be created.")
            sys.exit()


    def normal_store(self):
        """
        Stores a new outline of the story into memory, appending to existing memory.
        Uses the language model to process the new outline based on existing memory.
        :return: The updated memory_store content.
        """
        print ( f"Generating memory..." )
        try:
            system_message_prompt = SystemMessagePromptTemplate.from_template(SYS_MEMORY_PROMPT)
            human_message = HumanMessagePromptTemplate.from_template(WRITE_MEMORY_PROMPT)
            prompt_setting = ChatPromptTemplate.from_messages ( [system_message_prompt , human_message] )
            init_prompt = prompt_setting.format (
                topic = self.state['Topic'],
                main_character = self.state['MainCharacter'],
                main_goal = self.state['MainGoal'],
                language = self.state['Language'],
                new_outline = self.state["RecentStory"][-1],
                memory_storage = self.pull_memory(MEMORY_STORAGE_PATH)
            )
            response = self.llm.invoke ( [init_prompt] ).content
            memory = memory_parser(response)
            self.memory_store = memory
        except:
            warnings.warn(f"Memory store could not be created.")
            sys.exit()

    def pull_memory(self, path:str = MEMORY_STORAGE_PATH):
        """
        Retrieves the current content of the memory_store.
        :return: The content stored in memory.
        """
        try:
            print(f"Pulling memory...")
            with open(path,'r') as f:
                memory = f.read()
            return memory
        except:
            warnings.warn(f"Memory store could not be pulled.")


    def write_down_settings(self,path:str=STORY_SETTING_PATH):
        """
        Writes the story settings to a JSON file.
        :param path: The file path where the JSON data will be saved.
        """
        try:
            print(f"Writing settings to your story setting path...")
            with open(path,'w') as f:
                json.dump(self.state,f)
        except:
            warnings.warn(f"Memory store could not be written down.")

    def write_down_memory(self,path:str = MEMORY_STORAGE_PATH):
        """
        Writes the current memory_store content to a JSON file.
        :param path: The file path where the JSON data will be saved.
        """
        try:
            print ( f"Writing long-term memories to your story memory path..." )
            with open(path,'a') as f:
                json.dump(self.memory_store,f)
        except:
            warnings.warn(f"Memory store could not be written down.")

    def delete_memory(self,path:str = MEMORY_STORAGE_PATH):
        """
        Deletes the JSON file containing the memory_store content.
        :param path: The file path of the JSON file to be deleted.
        """
        os.remove(path)