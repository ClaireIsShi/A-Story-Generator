'''
-- @Time    : 2025/7/10 01:51
-- @File    : build.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

import os,sys
from StoryState import StoryState
from langgraph.constants import START , END
from langgraph.graph import StateGraph
import warnings
# path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TwistGenerator.KnowledgeGraphProcess import catch_nodes_of_original_story , generate_twist_for_outline

warnings.filterwarnings("ignore")
from utils import set_env
set_env()
Twist_subgraph = StateGraph(StoryState,output = StoryState)
Twist_subgraph.add_node('catch_nodes_of_original_story',catch_nodes_of_original_story)
Twist_subgraph.add_node('generate_twist_for_outline',generate_twist_for_outline)
Twist_subgraph.add_edge(START, 'catch_nodes_of_original_story')
Twist_subgraph.add_edge('catch_nodes_of_original_story', 'generate_twist_for_outline')
Twist_subgraph.add_edge('generate_twist_for_outline', END)