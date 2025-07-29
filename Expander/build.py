'''
-- @Time    : 2025/7/14 22:39
-- @File    : build.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
from Expander.Interact import calculate_similarity,generate_expansion, write_to_memory, clean_outline

'''
from Expander import generate_expansion, write_to_memory, clean_outline,calculate_similarity
state = generate_expansion ( state, length, write_to_json:Optional[str] = None )
state = calculate_similarity(state)
state = clean_outline(state)
state = write_to_memory(state)
connect as
generate_expansion --> calculate_similarity --> clean_outline --> write_to_memory
'''
from utils import set_env
set_env()
from langgraph.constants import START , END
from langgraph.graph import StateGraph
from StoryState import StoryState
from utils import set_env
set_env()
import warnings
warnings.filterwarnings("ignore")

Expender_subgraph = StateGraph(StoryState, output = StoryState)
Expender_subgraph.add_node('calculate_similarity',calculate_similarity)
Expender_subgraph.add_node('generate_expansion',generate_expansion)
Expender_subgraph.add_node('write_to_memory',write_to_memory)
Expender_subgraph.add_node('clean_outline',clean_outline)
Expender_subgraph.add_edge(START,'generate_expansion')
Expender_subgraph.add_edge('generate_expansion','calculate_similarity')
Expender_subgraph.add_edge('calculate_similarity','clean_outline')
Expender_subgraph.add_edge('clean_outline','write_to_memory')
Expender_subgraph.add_edge('write_to_memory', END)