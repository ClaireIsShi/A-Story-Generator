'''
-- @Time    : 2025/7/11 22:56
-- @File    : Interact.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
from Expender.ReaderSimulator import ReaderSimulator
from Expender.ExpenderWriterSimulator import ExpenderWriterSimulator
import os
from langchain_anthropic import ChatAnthropic
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-trained model for sentence embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import set_env, get_content_between_a_b
from StoryState import StoryState
from settings import EXPEND_LEN , FINAL_STORY_PATH
from Memory.MemoryStore import MemoryStore

# Prompt template for completing incomplete story endings
FINISH_SENTENCE_PROMPT = """
You're a good story writer and a native speaker of {language}. Now you get one part of your story:{story}. Your job is to complete the story and delete repetitive part.
Follow these steps below to edit this story part:
1. Read the story part carefully and delete repetitive part;
2. Check the last sentence of this part of story, if the last sentence is not complete, you should generate a new sentence to complete that one sentence and output the completed whole story. 
Output your story in this format:
## whole story:
<your new story>.
## END
"""


def get_whole_story(story: str):
    """
    Extract the complete story content from the formatted input string.
    :param story: (str) Input string containing the story in the specified format.
    :return: (str) The extracted whole story content.
    """
    return get_content_between_a_b('## whole story:', '## END', story)


# Set environment variables required for the application
set_env()


def interact(state: StoryState, length: int = EXPEND_LEN, llm=ChatAnthropic(model_name="claude-3-opus-20240229")):
    """
    Facilitates interaction between the story expander and reader simulator to generate story content.
    Handles both initial story generation (when StartSign is True) and subsequent expansions (when StartSign is False).
    :param state: (StoryState) Object containing current story state and metadata.
    :param length: (int) Target length for the generated story content (default from EXPEND_LEN).
    :param llm: (ChatAnthropic) Language model instance used for generation (default: claude-3-opus-20240229).
    :return: (tuple) Generated text content and updated StoryState object.
    """
    if state['StartSign']:
        # Initialize expander for the first story generation
        expender = ExpenderWriterSimulator(state, llm, length)
        initial_first_outline = expender.initial_first_outline()
        # Simulate reader feedback on the initial outline
        reader = ReaderSimulator(expender.state, initial_first_outline)
        logical, emotional, state = reader()
        # Generate expanded content based on reader feedback
        initial_second_outline = expender(logical, emotional)
        # Switch to non-initial mode for subsequent generations
        expender.set_startsign_to_false()
        # Generate final part of the initial story
        last_first_outline = expender.initial_last_task()
        reader = ReaderSimulator(expender.state, last_first_outline)
        logical, emotional, state = reader()
        last_second_outline = expender(logical, emotional)
        # Combine all parts for the initial full story
        text = initial_second_outline + last_second_outline
    else:
        # Generate subsequent story expansions (non-initial mode)
        expender = ExpenderWriterSimulator(state, llm, length)
        last_first_outline = expender.initial_last_task()
        reader = ReaderSimulator(expender.state, last_first_outline)
        logical, emotional, state = reader()
        last_second_outline = expender(logical, emotional)
        text = last_second_outline
    return text, state


# Core node function for story expansion
def generate_expansion(state: StoryState, length: int = EXPEND_LEN, write_to_json: Optional[str] = FINAL_STORY_PATH):
    """
    Generates expanded story content, updates the story state, and optionally saves to a file.
    :param state: (StoryState) Current story state.
    :param length: (int) Target length for the expansion (default from EXPEND_LEN).
    :param write_to_json: (Optional[str]) Path to save the generated content (default from FINAL_STORY_PATH).
    :return: (StoryState) Updated story state with new content and length.
    """
    final_generated, state = interact(state, length=length)
    # Ensure generated content is not empty
    assert len(final_generated) > 0, "The generated text is empty."
    # Update total story length in state
    state['TotalStoryLength'] += len(final_generated)
    # Save to file if path is provided, otherwise print
    if write_to_json:
        print(f"Saving story at your storage path...")
        with open(write_to_json, "a", encoding="utf-8") as f:
            f.write(final_generated)
    else:
        print(f"generating {len(final_generated)} words storyline:\n", final_generated)
    return state


def calculate_similarity(state):
    """
    Calculates cosine similarity between the two most recent story outlines in the state.
    Updates the state with the similarity score.
    :param state: (dict) Story state containing 'RecentStory' list.
    :return: (dict) Updated state with 'similarity' key.
    """
    # Extract recent story outlines from state
    recent_story = state['RecentStory']
    # Validate input format
    if not isinstance(recent_story, list):
        raise ValueError("recent_story must be a list")
    # Generate embeddings for the two most recent outlines
    emb1 = model.encode([recent_story[0]])
    emb2 = model.encode([recent_story[1]])
    # Compute cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    state['similarity'] = similarity
    print(similarity)
    return state


def clean_outline(state: StoryState) -> StoryState:
    """
    Cleans up the story outline by resetting StartSign and retaining only the latest story in RecentStory.
    :param state: (StoryState) Current story state.
    :return: (StoryState) Cleaned story state.
    """
    state['StartSign'] = False
    # Keep only the most recent story if there are multiple entries
    if len(state['RecentStory']) > 1:
        state['RecentStory'] = [state['RecentStory'][-1]]
    return state


def write_to_memory(state: StoryState) -> StoryState:
    """
    Saves the current story state to memory using MemoryStore.
    :param state: (StoryState) Current story state to be stored.
    :return: (StoryState) Updated story state after memory storage.
    """
    memory_store = MemoryStore(state)
    memory_store.normal_store()
    memory_store.write_down_memory()
    return state