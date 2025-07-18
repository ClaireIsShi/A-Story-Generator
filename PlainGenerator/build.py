'''
-- @Time    : 2025/7/10 18:51
-- @File    : build.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

from PlainGenerator.PlainGenerate import generate_plain_story, check_and_pass

from utils import set_env
set_env()

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from StoryState import StoryState

# Set up environment variables again (redundant but ensures configuration is applied)

import warnings
warnings.filterwarnings("ignore")



Plain_subgraph = StateGraph(StoryState)

Plain_subgraph.add_node('check_and_pass', check_and_pass)
Plain_subgraph.add_edge('check_and_pass', 'generate_plain_story')
Plain_subgraph.add_node('generate_plain_story', generate_plain_story)
Plain_subgraph.add_edge(START, 'check_and_pass')
Plain_subgraph.add_edge('generate_plain_story', END)