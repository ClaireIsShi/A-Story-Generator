'''
-- @Time    : 2025/7/9 19:33
-- @File    : KnowledgeGraphProcess.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
# Import necessary system and OS modules
import os,sys
# Import type hints for type checking
from typing import TypedDict , List
# Import ChatOpenAI from langchain_openai for language model interactions
from langchain_openai import ChatOpenAI
# Import StoryState class for managing story-related states
from StoryState import StoryState

# Import the warnings module to handle warnings
import warnings

# Import functions for twist processing and abstract extraction
from TwistGenerator.SimilaityCalculate import process_twist,get_abstract
# Import the utility language model from settings
from settings import UTIL_LLM

# Suppress all warnings
warnings.filterwarnings("ignore")
# Import utility functions and set environment variables
from utils import get_content_between_a_b,set_env
set_env()

# Add the parent directory to the system path to allow module imports
# Get the current working directory
current_dir = os.getcwd()
# Get the parent directory of the current working directory
parent_dir = os.path.dirname(current_dir)
# Insert the parent directory at the beginning of the system path
sys.path.insert(0, parent_dir)

# Set a global variable for the length parameter
LENGTH = 500

class TwistKG(TypedDict, total=False):
    """
    Defines an optional typed dictionary for storing knowledge graph information related to story twists.
    """
    MainCharacter: str  # Name of the main character in the story
    MainGoal: str  # Main goal of the main character in the story
    StartSign: bool  # Starting flag of the story
    RecentStory: List[str]  # List of recent story content
    Language: str  # Language used in the story
    Topic: str  # Topic of the story
    similarity: float  # Similarity value
    OriginalKG: str  # Original knowledge graph of the story
    TotalStoryLength: int  # Total length of the story

# Function to catch nodes of the original story
def catch_nodes_of_original_story(state: StoryState,llm=UTIL_LLM) -> TwistKG:
    print("Setting up TwistWritingAssistant...")
    print("Start to catch KG nodes in generated outline...")
    """
    Captures key nodes of the original story and generates knowledge graph information including the original knowledge graph.

    Args:
        state (StoryState): A state object containing story information.
        llm (ChatOpenAI): The language model used to generate the knowledge graph, defaulting to gpt-3.5-turbo.

    Returns:
        TwistKG: Knowledge graph information including the original knowledge graph.
    """
    # Number of attempts
    trying = 0
    # Flag to control the loop
    button = True
    while (trying < 4
           and button):
        try:
            # Get the abstract of the last recent story
            KG = get_abstract(state["RecentStory"][-1], state["Language"], llm=llm)
            # Set the flag to False to exit the loop
            button = False
        except:
            # Increment the number of attempts if an exception occurs
            trying += 1
    if trying == 4:
        print("Failed to catch nodes of original story.")
        # Exit the program if all attempts fail
        sys.exit()
    return {
        **state,
        "OriginalKG": KG
    }

# Function to generate a twist for the story outline
def generate_twist_for_outline(state:TwistKG) -> StoryState:
    """
    Generates twist content for the story outline.

    Args:
        state (TwistKG): An object containing knowledge graph information related to story twists.

    Returns:
        StoryState: An updated story state object.
    """
    # Generate the twist outline
    outline = process_twist(state["Language"], state["Topic"], state["OriginalKG"])
    print("Generating key twist nodes in generated outline...")
    if len(outline)>51:
        print('Outline in generated twist outline:',outline[:50],'...(etc.)')
    else:
        print('Outline in generated twist outline:',outline)
    return {
        "Language": state["Language"],
        "Topic": state["Topic"],
        "RecentStory":state['RecentStory']+[outline],
        "MainGoal": state["MainGoal"],
        "MainCharacter": state["MainCharacter"],
        "StartSign": state["StartSign"],
        "similarity": state["similarity"],
        "TotalStoryLength": state["TotalStoryLength"]
    }