'''
-- @Time    : 2025/7/14 19:37
-- @File    : settings.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
from typing import Optional
from langchain_anthropic import ChatAnthropic
import os
from langchain_openai import ChatOpenAI

current_dir = os.getcwd()
EXPEND_LEN = 700# best set is 3500 for English
SIMILARITY_THRESHOLD = 0.8
WRITE_TO_FILE: Optional[bool] = False
MAX_LEN = 20000

STORY_SETTING_PATH = current_dir + "/memory_storage/story_setting.json"
MEMORY_STORAGE_PATH = current_dir + "/memory_storage/memory.json"
FINAL_STORY_PATH = current_dir + "/result.json"
# which LLM to expand story
WRITE_LLM = ChatAnthropic(model = 'claude-3-7-sonnet-20250219')
# which LLM to use as utils
UTIL_LLM = ChatOpenAI(model = 'gpt-3.5-turbo')