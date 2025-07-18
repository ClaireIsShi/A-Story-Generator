'''
-- @Time    : 2025/7/15 03:33
-- @File    : build.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
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
from End.EndsGenerate import end_generation

End_subgraph = StateGraph(StoryState)
End_subgraph.add_node('end_generation',end_generation)
End_subgraph.add_edge(START,'end_generation')
End_subgraph.add_edge('end_generation',END)