'''
-- @Time    : 2025/7/10 18:12
-- @File    : PlainGenerate.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
from StoryState import StoryState
from utils import set_env
set_env()
import warnings
warnings.filterwarnings("ignore")
from PlainGenerator.PlainWritingAssistant import PlainWritingAssistant


def generate_plain_story(state: StoryState, length:int = 400, long_term_memory:str = "")->StoryState:
    """
    Generates a plain story outline based on the given story state and parameters.
    Continuously attempts to initialize the writing assistant until successful, then generates the story.

    :param state: (StoryState) Object containing current story metadata (characters, goal, language, etc.)
    :param length: (int) Target length of the generated story segment (default: 400)
    :param long_term_memory: (str) Long-term memory context to guide the generation (default: empty string)
    :return: (StoryState) Updated story state with the new generated content added to RecentStory
    """
    # Create a PlainWritingAssistant instance
    button: bool = True
    while button:
        try:
            writing_assistant = PlainWritingAssistant(
                language=state['Language'],
                length=length,
                topic=state['Topic'],
                last_outline=state['RecentStory'][-1],
                goal=state['MainGoal'],
                long_term_memory=long_term_memory,
                start_sign=state['StartSign']
            )
            button = False
        except:
            continue
    # Generate the plain story
    chosen_outline = writing_assistant()
    state_final = {
        'MainCharacter': state['MainCharacter'],
        'MainGoal': state['MainGoal'],
        'StartSign': state['StartSign'],
        'RecentStory': state['RecentStory'] + chosen_outline.split("\n"),
        'Language': state['Language'],
        'Topic': state['Topic'],
        'similarity': 0,
        'TotalStoryLength': state['TotalStoryLength']
    }
    return state_final


def check_and_pass(state: StoryState) -> StoryState:
    """
    Validates and cleans the story state, ensuring consistency in the RecentStory field.
    If the story has started (StartSign is False) and RecentStory has multiple entries, retains only the latest one.

    :param state: (StoryState) Current story state to be checked and cleaned
    :return: (StoryState) Cleaned story state with updated RecentStory
    """
    if state['StartSign'] is False and len(state['RecentStory']) != 1:
        state['RecentStory'] = [state['RecentStory'][-1]]
    return state