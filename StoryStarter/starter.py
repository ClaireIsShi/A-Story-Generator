
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
You are a story creator, also a native speaker of {language}.
Tell me a beginning outline of a long story about {topic} in {language}. Your outline should be over 400 words.
You need to:
1. Character: Create a main character of the story.
2. Goal: Create a main goal for the main character; the goal should be close to the story topic.
3. Story: Create the beginning and setting of the long story, including the main character and their primary goal. The main goal can't be easy to reach.
Organize your output by strictly following the output format as below, don't change the format, English, such as "## main character:":
## main character:
<introduction of the main character in one sentence>
## main goal:
<introduction of the main character's main goal in one sentence>
## outline:
<put a "**This is the beginning of a long story**"sign next to the outline of the long story's beginning, which should be very close to the story topic>
## END
'''
START_WITH_MAIN_PROMPT = """
You are a story creator, also a native speaker in {language}.
Tell me a beginning outline of a long story about {topic} in {language}, your outline should be over 400 words.
You need to:
1. Setting a main character of the story: {main_character}, and the main goal of the story: {main_goal}, should be close to the story topic.
2. Write a Story Beginning Outline: create the long story's beginning and setting about the main character and the main goal. The main goal can't be easy to reach.
Organize your output by strictly following the output format as below, don't change the format, English, such as "## outline:":
## outline:
<put a "**This is the beginning of a long story**"sign next to the outline of the long story's beginning, should be very close to the story topic. >
## END
Here's a one-shot: your output should be like this:
## outline:
**This is the beginning of a long story** Allen is a young man who has been in love with Mika ever since they were children. They grew up together, sharing countless memories and experiences that have only deepened Eren's feelings for her. Eren's main goal is to final...(etc.)
## END
"""


# parser
def get_main_character(prompt):
    try:
        result = get_content_between_a_b('## main character:','## main goal:',prompt)
        if not result:
            warnings.warn("[错误位置: get_main_character] 无法从提示词中提取主要角色信息，请检查 LLM 返回格式是否正确")
        return result
    except Exception as e:
        warnings.warn(f"[错误位置: get_main_character] 提取主要角色时发生异常: {str(e)}")
        return None

def get_main_goal(prompt):
    try:
        result = get_content_between_a_b('## main goal:','## outline:',prompt)
        if not result:
            warnings.warn("[错误位置: get_main_goal] 无法从提示词中提取主要目标信息，请检查 LLM 返回格式是否正确")
        return result
    except Exception as e:
        warnings.warn(f"[错误位置: get_main_goal] 提取主要目标时发生异常: {str(e)}")
        return None

def get_outline(prompt):
    try:
        result = get_content_between_a_b('## outline:','## END', prompt)
        if not result:
            warnings.warn("[错误位置: get_outline] 无法从提示词中提取故事大纲信息，请检查 LLM 返回格式是否正确")
        return result
    except Exception as e:
        warnings.warn(f"[错误位置: get_outline] 提取故事大纲时发生异常: {str(e)}")
        return None



from settings import UTIL_LLM
llm = UTIL_LLM
# Node

def check_keys(state: dict):
    try:
        valid_keys = {"Language" , "Topic"}
        all_keys = valid_keys.union ( {"MainCharacter" , "MainGoal"} )
        state_keys = set(state.keys())

        if state_keys != valid_keys and state_keys != all_keys:
            warnings.warn(
                f"[错误位置: check_keys] 输入状态字典的键不正确。\n"
                f"输入必须包含以下两种情况之一：\n"
                f"1. 仅包含 'Language' 和 'Topic'\n"
                f"2. 包含所有四个键: 'Language', 'Topic', 'MainCharacter', 'MainGoal'\n"
                f"当前输入的键: {list(state_keys)}"
            )
            import sys
            sys.exit ( 1 )
        return state
    except Exception as e:
        warnings.warn(f"[错误位置: check_keys] 检查状态字典键时发生异常: {str(e)}")
        import sys
        sys.exit(1)
def clean_dict(state: dict):
    try:
        if 'Language' not in state or 'Topic' not in state:
            warnings.warn(
                f"[错误位置: clean_dict] 状态字典缺少必需的键 'Language' 或 'Topic'。\n"
                f"当前输入的键: {list(state.keys())}"
            )
            return None
        return {
            'Language': state['Language'],
            "Topic": state["Topic"]
        }
    except Exception as e:
        warnings.warn(f"[错误位置: clean_dict] 清理状态字典时发生异常: {str(e)}")
        return None

def setting_of_story(state:StoryState)-> StoryState:
    print("Setting up StoryStarterBeginning...")
    def _set_story(state):
        try:
            if 'Language' not in state or 'Topic' not in state:
                warnings.warn(
                    f"[错误位置: setting_of_story._set_story] 状态字典缺少必需的键 'Language' 或 'Topic'。\n"
                    f"当前输入的键: {list(state.keys())}"
                )
                return None
            
            prompt = START_PRMPT.format ( language=state['Language'] , topic=state['Topic'] )
            try:
                response = llm.invoke ( prompt ).content
            except Exception as e:
                warnings.warn(
                    f"[错误位置: setting_of_story._set_story] 调用 LLM 生成故事设置时失败: {str(e)}\n"
                    f"请检查网络连接、API 密钥或 LLM 配置"
                )
                return None
            
            main_goal = get_main_goal ( response )
            main_character = get_main_character ( response )
            outline = get_outline ( response )
            
            if not main_goal or not main_character or not outline:
                warnings.warn(
                    f"[错误位置: setting_of_story._set_story] 从 LLM 响应中解析内容失败。\n"
                    f"主要目标: {'已提取' if main_goal else '未提取'}\n"
                    f"主要角色: {'已提取' if main_character else '未提取'}\n"
                    f"故事大纲: {'已提取' if outline else '未提取'}\n"
                    f"请检查 LLM 返回格式是否符合预期"
                )
                return None
            
            return {
                'Topic': state['Topic'] ,
                'Language': state['Language'] ,
                'MainGoal': main_goal ,
                'MainCharacter': main_character ,
                'RecentStory': [outline] ,
                'StartSign': True ,
                'similarity': 0,
            }
        except KeyError as e:
            warnings.warn(
                f"[错误位置: setting_of_story._set_story] 状态字典缺少必需的键: {str(e)}\n"
                f"当前输入的键: {list(state.keys()) if isinstance(state, dict) else '非字典类型'}"
            )
            return None
        except Exception as e:
            warnings.warn(
                f"[错误位置: setting_of_story._set_story] 设置故事时发生未知异常: {str(e)}\n"
                f"输入状态: {state}"
            )
            return None

    if state.get('MainCharacter') is not None:
        try:
            if 'Language' not in state or 'Topic' not in state:
                warnings.warn(
                    f"[错误位置: setting_of_story (已有角色模式)] 状态字典缺少必需的键 'Language' 或 'Topic'。\n"
                    f"当前输入的键: {list(state.keys())}"
                )
                return None
            
            MainGoal = state.get ( 'MainGoal' )
            if not MainGoal:
                warnings.warn(
                    f"[错误位置: setting_of_story (已有角色模式)] 状态字典中 'MainGoal' 为空或不存在。\n"
                    f"当前输入的键: {list(state.keys())}"
                )
                return None
            
            try:
                p = START_WITH_MAIN_PROMPT.format ( 
                    language=state['Language'] , 
                    topic=state['Topic'] ,
                    main_character=state['MainCharacter'],
                    main_goal=MainGoal
                )
            except KeyError as e:
                warnings.warn(
                    f"[错误位置: setting_of_story (已有角色模式)] 格式化提示词时缺少必需的键: {str(e)}\n"
                    f"当前输入的键: {list(state.keys())}"
                )
                return None
            
            try:
                response = llm.invoke ( p ).content
            except Exception as e:
                warnings.warn(
                    f"[错误位置: setting_of_story (已有角色模式)] 调用 LLM 生成故事大纲时失败: {str(e)}\n"
                    f"请检查网络连接、API 密钥或 LLM 配置"
                )
                return None
            
            outline = get_outline ( response )
            if not outline:
                warnings.warn(
                    f"[错误位置: setting_of_story (已有角色模式)] 从 LLM 响应中提取故事大纲失败。\n"
                    f"请检查 LLM 返回格式是否符合预期"
                )
                return None
            
            state ['RecentStory'] =[outline]
            state['similarity'] = 0
            state['StartSign'] = True
            state['TotalStoryLength'] = 0
            return state
        except Exception as e:
            warnings.warn(
                f"[错误位置: setting_of_story (已有角色模式)] 处理已有角色和目标的输入时发生异常: {str(e)}\n"
                f"输入要求: 字典应包含键 'Language', 'Topic', 'MainGoal', 'MainCharacter'\n"
                f"或仅包含键 'Language', 'Topic'\n"
                f"当前输入: {state}"
            )
            return None
    if state.get('MainCharacter') is None and state.get('MainGoal') is None:
        return _set_story(state)


def judge_if_similarity_higher_enough(state:StoryState) -> bool:
    try:
        if 'MainCharacter' not in state or 'MainGoal' not in state or 'Topic' not in state:
            warnings.warn(
                f"[错误位置: judge_if_similarity_higher_enough] 状态字典缺少必需的键。\n"
                f"需要包含: 'MainCharacter', 'MainGoal', 'Topic'\n"
                f"当前输入的键: {list(state.keys())}"
            )
            return False
        
        try:
            similarity_result = get_similarity([state['MainCharacter'] , state['MainGoal']])
            if not similarity_result or len(similarity_result) < 2:
                warnings.warn(
                    f"[错误位置: judge_if_similarity_higher_enough] 计算主要角色与目标的相似度时返回结果格式不正确\n"
                    f"返回结果: {similarity_result}"
                )
                return False
            similarity_beginning = similarity_result[1]
        except Exception as e:
            warnings.warn(
                f"[错误位置: judge_if_similarity_higher_enough] 计算主要角色与目标的相似度时发生异常: {str(e)}"
            )
            return False
        
        try:
            similarity_result = get_similarity([state['Topic'] , state['MainGoal']])
            if not similarity_result or len(similarity_result) < 2:
                warnings.warn(
                    f"[错误位置: judge_if_similarity_higher_enough] 计算主题与目标的相似度时返回结果格式不正确\n"
                    f"返回结果: {similarity_result}"
                )
                return False
            similarity_topic = similarity_result[1]
        except Exception as e:
            warnings.warn(
                f"[错误位置: judge_if_similarity_higher_enough] 计算主题与目标的相似度时发生异常: {str(e)}"
            )
            return False
        
        if 'Language' not in state:
            warnings.warn(
                f"[错误位置: judge_if_similarity_higher_enough] 状态字典缺少 'Language' 键，无法判断语言类型\n"
                f"当前输入的键: {list(state.keys())}"
            )
            return False
        
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
    except KeyError as e:
        warnings.warn(
            f"[错误位置: judge_if_similarity_higher_enough] 状态字典缺少必需的键: {str(e)}\n"
            f"输入要求: 字典应包含键 'Language', 'Topic', 'MainGoal', 'MainCharacter'\n"
            f"或仅包含键 'Language', 'Topic'\n"
            f"当前输入: {state}"
        )
        return False
    except Exception as e:
        warnings.warn(
            f"[错误位置: judge_if_similarity_higher_enough] 判断相似度时发生未知异常: {str(e)}\n"
            f"输入状态: {state}"
        )
        return False

from memory_storage.MemoryStore import MemoryStore
def store_to_memory(state:StoryState) -> StoryState:
    try:
        if not state:
            warnings.warn(
                f"[错误位置: store_to_memory] 输入状态为空，无法存储到内存"
            )
            return state
        
        try:
            memory_store = MemoryStore(state)
        except Exception as e:
            warnings.warn(
                f"[错误位置: store_to_memory] 创建 MemoryStore 实例时失败: {str(e)}\n"
                f"请检查 MemoryStore 类的初始化和状态字典格式"
            )
            return state
        
        try:
            memory_store.first_store()
        except Exception as e:
            warnings.warn(
                f"[错误位置: store_to_memory] 执行 first_store() 时失败: {str(e)}"
            )
        
        try:
            memory_store.write_down_settings()
        except Exception as e:
            warnings.warn(
                f"[错误位置: store_to_memory] 写入故事设置时失败: {str(e)}"
            )
        
        try:
            memory_store.write_down_memory()
        except Exception as e:
            warnings.warn(
                f"[错误位置: store_to_memory] 写入记忆时失败: {str(e)}"
            )
        
        return {
            **state,  # 保留原状态中的所有键值对
        }
    except Exception as e:
        warnings.warn(
            f"[错误位置: store_to_memory] 存储到内存时发生未知异常: {str(e)}\n"
            f"输入状态: {state}"
        )
        return state

def judge_if_set_Main_by_user(state:StoryState) -> bool:
    try:
        if not isinstance(state, dict):
            warnings.warn(
                f"[错误位置: judge_if_set_Main_by_user] 输入不是字典类型: {type(state)}"
            )
            return False
        
        if state.get('MainCharacter') is None and state.get('MainGoal') is None:
            return False
        else:
            return True
    except Exception as e:
        warnings.warn(
            f"[错误位置: judge_if_set_Main_by_user] 判断用户是否设置主要角色和目标时发生异常: {str(e)}\n"
            f"输入状态: {state}"
        )
        return False