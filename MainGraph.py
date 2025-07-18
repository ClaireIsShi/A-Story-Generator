import os , sys
from utils import set_env

set_env ()
current_dir = os.getcwd ()
parent_dir = os.path.dirname ( current_dir )
# Add the parent directory to sys.path for module imports
sys.path.insert ( 0 , parent_dir )
from StoryStarter import Starter_subgraph
from StoryState import StoryState
from TwistGenerator import Twist_subgraph
from PlainGenerator import Plain_subgraph
from langgraph.constants import START , END
from langgraph.graph import StateGraph
from Expender import Expender_subgraph
from End import End_subgraph

# Add the parent directory to sys.path again (redundant but ensures configuration)
from settings import *

# Compile subgraphs into executable nodes for the main graph
twist_subgraph = Twist_subgraph.compile ()
starter_subgraph = Starter_subgraph.compile ()
plain_subgraph = Plain_subgraph.compile ()
expender_subgraph = Expender_subgraph.compile ()
end_subgraph = End_subgraph.compile ()


# Decision function for conditional edge routing
# Determines if the story's similarity score exceeds the threshold
def if_similarity_higher_than_threshold(state: StoryState , similarity_threshold=SIMILARITY_THRESHOLD) -> bool:
    """
    Checks if the story's current similarity score meets or exceeds the defined threshold.
    High similarity indicates repetitive content, triggering a plot twist.

    :param state: (StoryState) Current story state containing similarity metric.
    :param similarity_threshold: (float) Threshold for determining when a twist is needed (default from settings).
    :return: (bool) True if similarity is high, False otherwise.
    """
    if state['similarity'] >= similarity_threshold:
        return True
    else:
        return False


# Decision function for stopping story generation
# Determines if the story has reached the maximum desired length
def if_stop_generate(state: StoryState , max_length=MAX_LEN) -> bool:
    """
    Checks if the total generated story length has reached the predefined maximum.

    :param state: (StoryState) Current story state containing length metric.
    :param max_length: (int) Maximum allowed story length (default from settings).
    :return: (bool) True if maximum length is reached, False otherwise.
    """
    if state['TotalStoryLength'] >= max_length:
        return True
    else:
        print ( "One generate round is finished. Now start to generate next round." )
        return False


# Passthrough node that returns the state unchanged
# Used to maintain graph flow without modifying state
def pass_node(state: StoryState) -> StoryState:
    """
    Simple passthrough function that returns the state without modification.
    Used in the graph to route flow without altering the story state.

    :param state: (StoryState) Input story state.
    :return: (StoryState) Unchanged story state.
    """
    return state


# Construct the main story generation graph using compiled subgraphs and decision nodes
# See test.ipynb for visual path documentation
MainGraph = StateGraph ( StoryState )
MainGraph.add_node ( "StoryStarter" , starter_subgraph )  # Initial story setup
MainGraph.add_node ( "TwistGenerator" , twist_subgraph )  # Generates plot twists
MainGraph.add_node ( "PlainGenerator" , plain_subgraph )  # Generates standard story segments
MainGraph.add_node ( "Expander" , expender_subgraph )  # Expands and refines story content
MainGraph.add_node ( "Pass_condition" , pass_node )  # Passthrough for conditional routing
MainGraph.add_node ( "End" , end_subgraph )  # Generates story endings

# Define edges to create the story generation workflow
MainGraph.add_edge ( START , "StoryStarter" )  # Workflow begins with story setup
MainGraph.add_edge ( "StoryStarter" , "PlainGenerator" )  # After setup, generate initial plot twist
MainGraph.add_edge ( "PlainGenerator" , "Expander" )  # Route plain segments to expander
MainGraph.add_edge ( "TwistGenerator" , "Expander" )  # Route twists to expander

# Conditional edge: decide whether to end the story or continue
MainGraph.add_conditional_edges ( "Expander" ,
                                  if_stop_generate , {
                                      True: 'End' ,  # If max length reached, generate ending
                                      False: 'Pass_condition'  # Otherwise, check similarity
                                  } )
# Conditional edge: decide between generating a twist or plain segment
MainGraph.add_conditional_edges ( "Pass_condition" ,
                                  if_similarity_higher_than_threshold , {
                                      True: 'TwistGenerator' ,  # If similarity is high, introduce a twist
                                      False: "PlainGenerator"  # If similarity is low, continue with plain content
                                  } )
MainGraph.add_edge ( "End" , END )  # Workflow ends after generating the ending
# Compile the main graph into an executable form
main_graph = MainGraph.compile ()