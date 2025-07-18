from langgraph.constants import START , END
from langgraph.graph import StateGraph
import os

from StoryState import StoryState

from StoryStarter.starter import *
import warnings
warnings.filterwarnings("ignore")
little_graph = StateGraph(StoryState, output = StoryState)
little_graph.add_node("setting_of_story", setting_of_story)
little_graph.add_edge(START, "setting_of_story")
little_graph.add_node('clean_dict',clean_dict)
little_graph.add_edge("clean_dict", "setting_of_story")
little_graph.add_conditional_edges(
    'setting_of_story',
    judge_if_similarity_higher_enough,
    {True: END,
     False: 'clean_dict'}
)

Starter_subgraph = StateGraph(StoryState, output = StoryState)
Starter_subgraph.add_node("setting_of_story", setting_of_story)
Starter_subgraph.add_node("store_to_memory", store_to_memory)
Starter_subgraph.add_node('little_graph', little_graph.compile())
Starter_subgraph.add_node('check_keys', check_keys)
Starter_subgraph.add_edge(START, 'check_keys')
Starter_subgraph.add_conditional_edges("check_keys",
                                       judge_if_set_Main_by_user,
                                       {
                                           True: "setting_of_story",
                                           False:'little_graph'
                                       }
                                       )
Starter_subgraph.add_edge('little_graph', "store_to_memory")
Starter_subgraph.add_edge("setting_of_story", "store_to_memory")
Starter_subgraph.add_edge("store_to_memory", END)






'''
Starter_subgraph = StateGraph(StoryStarterBeginning, output = StoryStarterEnd)
Starter_subgraph.add_edge(START, "setting_of_story")
Starter_subgraph.add_node("setting_of_story", setting_of_story)
Starter_subgraph.add_edge("setting_of_story", END)
Starter_subgraph.add_node("pass_node_state", pass_INFO)
Starter_subgraph.add_node("store_to_memory", store_to_memory)

Starter_subgraph.add_conditional_edges(
        "setting_of_story",
        judge_if_similarity_higher_enough,
        {
            True: "store_to_memory",
            False: "setting_of_story",
        }
    )
Starter_subgraph.add_edge("store_to_memory", END)
'''