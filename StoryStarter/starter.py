
import sys
import os
from typing import TypedDict , Dict

from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings("ignore")

# 获取当前脚本所在目录的上一级目录
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到 sys.path 中
sys.path.insert(0, parent_dir)
from utils import get_content_between_a_b, set_env,get_similarity
set_env()
from StoryState import StoryState
START_PRMPT='''
Imagine you are a story creator, also a native speaker in {language}.
Tell me a beginning outline of a long story about {topic} in your own language, writing in your own language, your outline should be over 400 words.
You need to:
1. Character: create a main character of the story.
2. Goal: create a main goal for the main character, the goal should be close to the story topic.
3. Story: create the long story's beginning and setting about the main character and the main goal. The main goal can't be easy to reach.
organize your output by strictly following the output format as below, don't change the format English, such as "## main character:":
## main character:
<introduction of the main character in one sentence>
## main goal:
<introduction of the main character's main goal in one sentence>
## outline:
<put a "**This is the beginning of a long story**"sign next to the outline of the long story's beginning, should be very close to the story topic>
## END
'''
START_WITH_MAIN_PROMPT = """
Imagine you are a story creator, also a native speaker in {language}.
Tell me a beginning outline of a long story about {topic} in your own language, writing in your own language, your outline should be over 400 words.
You need to:
1. Setting a main character of the story: {main_character}, and the main goal of the story: {main_goal}, should be close to the story topic.
2. write a Story Beginning Outline: create the long story's beginning and setting about the main character and the main goal. The main goal can't be easy to reach.
organize your output by strictly following the output format as below, don't change the format English, such as "## outline:":
## outline:
<put a "**This is the beginning of a long story**"sign next to the outline of the long story's beginning, should be very close to the story topic. >
## END
here's one-shot: your output should be like this:
## outline:
**This is the beginning of a long story**Eren is a young man who has been in love with Mikasa ever since they were children. They grew up together, sharing countless memories and experiences that have only deepened Eren's feelings for her. Eren's main goal is to final...(etc.)
## END
"""


# parser
def get_main_character(prompt):
    return get_content_between_a_b('## main character:','## main goal:',prompt)
def get_main_goal(prompt):
    return get_content_between_a_b('## main goal:','## outline:',prompt)
def get_outline(prompt):
    return get_content_between_a_b('## outline:','## END', prompt)



from settings import UTIL_LLM
llm = UTIL_LLM
# Node

def check_keys(state: dict):
    valid_keys = {"Language" , "Topic"}
    all_keys = valid_keys.union ( {"MainCharacter" , "MainGoal"} )
    state_keys = state.keys()

    if state_keys not in (valid_keys , all_keys):
        import warnings
        warnings.warn (
            "Input error: The input must contain either 'Language' and 'Topic' only, or all four keys: 'Language', 'Topic', 'MainCharacter', 'MainGoal'." )
        import sys
        sys.exit ( 1 )
    return state
def clean_dict(state: dict):
    return {
        'Language': state['Language'],
        "Topic": state["Topic"]
    }

def setting_of_story(state:StoryState)-> StoryState:
    print("Setting up StoryStarterBeginning...")
    def _set_story(state):
        prompt = START_PRMPT.format ( language=state['Language'] , topic=state['Topic'] )
        try:
            response = llm.invoke ( prompt ).content
            main_goal = get_main_goal ( response )
            main_character = get_main_character ( response )
            outline = get_outline ( response )
            return {
                'Topic': state['Topic'] ,
                'Language': state['Language'] ,
                'MainGoal': main_goal ,
                'MainCharacter': main_character ,
                'RecentStory': [outline] ,
                'StartSign': True ,
                'similarity': 0,
            }
        except:
            warnings.warn (
                f"Error in StoryStarter: setting_of_story, please check your input.\nYour input is: {state}" )
            return None

    if state.get('MainCharacter') is not None:
        try:
            MainGoal = state.get ( 'MainGoal' )
            p = START_WITH_MAIN_PROMPT.format ( language=state['Language'] , topic=state['Topic'] ,main_character=state['MainCharacter'],main_goal=MainGoal)
            response = llm.invoke ( p ).content
            outline = get_outline ( response )
            state ['RecentStory'] =[outline]
            state['similarity'] = 0
            state['StartSign'] = True
            state['TotalStoryLength'] = 0
            return state
        except:
            warnings.warn(f"Input error:\n"
                          f"Your input can be Dict with keys: 'Language', 'Topic', 'MainGoal', 'MainCharacter'.\n"
                          f"or Dict with keys:'Language', 'Topic'\n"
                          f"But Your input is: {state}\n")
            return None
    if state.get('MainCharacter') is None and state.get('MainGoal') is None:
        return _set_story(state)


def judge_if_similarity_higher_enough(state:StoryState) -> bool:
    try:
        similarity_beginning = get_similarity (
                [state['MainCharacter'] , state['MainGoal']]
            )[1]
        similarity_topic = get_similarity (
                [state['Topic'] , state['MainGoal']]
            )[1]
        if state['Language'].lower() == 'english':
            if similarity_beginning > 0.65 and similarity_topic > 0.15:
                return True
            else:
                return False
        else:
            if similarity_beginning > 0.65 and similarity_topic*10 > 0.15:
                return True
            else:
                return False
    except:
        warnings.warn(f"Error in StoryStarter: \nInput error:\n"
                          f"Your input can be Dict with keys: 'Language', 'Topic', 'MainGoal', 'MainCharacter'.\n"
                          f"or Dict with keys:'Language', 'Topic'\n"
                      f"Your input is: {state}")
        return False

from Memory.MemoryStore import MemoryStore
def store_to_memory(state:StoryState) -> StoryState:
    memory_store = MemoryStore(state)
    memory_store.first_store()
    memory_store.write_down_settings()
    memory_store.write_down_memory()
    return {
        **state,  # 保留原状态中的所有键值对
    }

def judge_if_set_Main_by_user(state:StoryState) -> bool:
    if state.get('MainCharacter') is None and state.get('MainGoal') is None:
        return False
    else:
        return True