'''
-- @Time    : 2025/7/15 03:14
-- @File    : EndsGenerate.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
import os, sys
from StoryState import StoryState
import warnings

warnings.filterwarnings("ignore")
from utils import get_content_between_a_b, set_env
set_env()
from settings import *

# Add the parent directory to sys.path for module imports
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Prompt template for generating story endings
END_PROMPT = '''
You are a story writer, also a native speaker of {language}.Here's a story needs an ending.
The former story's outline is: {outline}. And the last story is: {specific_story}.
Here's some basic information about the story: main character is {main_character}, main goal is {main_goal}, topic is {topic}.
now please write a good ending for this story.
Follow the format:
## ending:
<put your ending text here>
##END
'''

def parser_end(str):
    """
    Parses the generated response to extract the story ending content.
    Extracts text between the markers "## ending:" and "##END".

    :param str: (str) Raw response string containing the formatted ending.
    :return: (str) Extracted ending text.
    """
    return get_content_between_a_b("## ending:", "##END", str)

def pull_long_story(path = MEMORY_STORAGE_PATH):
    """
    Reads the full story content from the memory storage file.

    :param path: (str) Path to the memory storage file (default from settings.MEMORY_STORAGE_PATH).
    :return: (str) Full story content stored in the file.
    """
    with open(path, "r") as f:
        story = f.read()
    return story




def end_generation(story_state: StoryState)->StoryState:
    """
    Generates an ending for the story based on the current story state and saves it to the final output file.
    Uses a language model to generate the ending, with retries on failure (up to 3 attempts).

    :param story_state: (StoryState) Object containing story metadata (characters, goal, topic, etc.) and recent story outlines.
    :return: (StoryState) Updated story state after generating and saving the ending.
    """
    # Format the prompt with story details from the current state
    prompt = END_PROMPT.format(
        language=story_state['Language'],
        outline=story_state['RecentStory'],
        specific_story=pull_long_story(),
        main_character=story_state['MainCharacter'],
        main_goal=story_state['MainGoal'],
        topic=story_state['Topic']
    )

    button = True
    trying = 0
    # Attempt to generate the ending with up to 3 retries
    while button and trying < 4:
        try:
            # Invoke the language model to generate the ending
            res = WRITE_LLM.invoke(prompt).content
            # Parse the generated response to extract the ending
            end = parser_end(res)
            button = False
            print(f"Finally! The story's generation is finished.")
        except:
            trying += 1
            print("End Generation doesn't work, try again...")

    # Handle failure after 3 attempts
    if trying == 4:
        warnings.warn("The end generation failed.")
        sys.exit()

    # Append the generated ending to the final story file
    with open(FINAL_STORY_PATH, 'a', encoding='UTF-8') as f:
        f.write(end)
    print("Saved your story to file:", os.path.basename(FINAL_STORY_PATH))
    story_state['TotalStoryLength'] += len(end)
    return story_state