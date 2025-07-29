'''
-- @Time    : 2025/7/11 16:50
-- @File    : ReaderSimulator.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''
import os
import warnings
from typing import Tuple , Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os,sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import set_env,get_content_between_a_b
from StoryState import StoryState
from settings import UTIL_LLM, WRITE_LLM
# Set environment variables
set_env()

# System prompt
CHECK_SYS_PRMPT = """
You're a delicate and experienced {topic} story reader and a native speaker of {language}. You're a good thinker and eager to speak out about some issues in the story, and you also focus on the details of the story. 
"""
# Simulated human author's question
WRITER_ASK_PRMPT = """
I'm writing a story based on the following information:
topic: {topic}, Main character: {main_character}, Main Goal:{main_goal} language: {language}.
Now here's a part of my story: {story}.
Do you have any idea about the story? Follow these steps to give me your response:
1. You need to read this part of the story CAREFULLY;
2. Is there any part you find hard to logically understand? Give me your confusion and suggestions. If you can understand this part of the story well, then just give an empty input;
3. Do these details in this part of the story logically make sense? Is the character growth of {main_character} detailed enough? If not, give me your suggestion. Do you think the details are good enough for you to understand this part of the story? If you can understand character growth well, then just give me an empty response.
Give me your response in the following format:
## logical detail confusion:
<here, put your confusion and suggestion in the logic of the part story in {language} you find in step 2>
## Character growth confusion:
<here, put your confusion and suggestion in the character growth of the part story in {language} you find in step 3>
## END
"""

# Single LLM, rewrite prompt
REWRITE_PROMPT = """
You're a good story writer and a native speaker of {language}. Now you get one part of your story:{story}. The story is based on the following information:
topic: {topic}, Main character: {main_character}, Main Goal:{main_goal} language: {language}.
Your job now is to rewrite this story based on the following reader's suggestion:
logical detail confusion and suggestion: {logical_confusion_and_suggestion}
main character of this story's character growth confusion: {character_growth_confusion_and_suggestion}
"""
from settings import WRITE_LLM
class ReaderSimulator:
    def __init__(self, state:StoryState, text:str, llm = WRITE_LLM):
        """
        Initialize an instance of the ReaderSimulator class.

        :param state: (StoryState) Object containing story metadata (topic, main characters, goal, language).
        :param text: (str) Segment of the story to be evaluated by the reader.
        :param llm: (ChatAnthropic) Language model instance for generating feedback (default: claude-sonnet-4-20250514).
        """
        self.state = state
        self.text = text
        self.llm = llm
        self.topic = self.state['Topic']
        self.main_character = self.state['MainCharacter']
        self.main_goal = self.state['MainGoal']
        self.language = self.state['Language']

    def set_sys(self):
        """
        Configure the system prompt and create a language model chain for reader simulation.

        :return: (Chain) LangChain chain combining the system prompt and LLM, or None if configuration fails.
        """
        prompt = ChatPromptTemplate.from_messages (
            [
                # System prompt defining the reader's role and context
                ("system" , CHECK_SYS_PRMPT.format(
                    topic = self.topic,
                    language = self.language
                )) ,
                # User prompt for inputting the story segment
                ("user" , "{input}"
                ),
            ]
        )
        try:
            # Create a chain by combining the prompt with the LLM
            chain = prompt | self.llm

        except:
            warnings.warn("The prompt is invalid. Check LangChain or graph configuration.")
            return None
        return chain

    def run(self):
        """
        Execute the LLM chain to generate feedback on the story segment.

        :return: (str) Feedback content from the LLM, or None if execution fails.
        """
        chain = self.set_sys()
        try:
            # Invoke the chain with the story segment and metadata
            self.response = chain.invoke(
                {
                    "input":
                        WRITER_ASK_PRMPT.format(
                            topic = self.topic,
                            main_character = self.main_character,
                            main_goal = self.main_goal,
                            language = self.language,
                            story = self.text
                        )
                    }
            ).content
        except:
            return None
        return self.response

    def response_parser(self):
        """
        Parse the LLM's feedback to extract logical and emotional critiques.
        Uses helper function to extract content between predefined markers.
        """
        response = self.run()
        if response:
            # Extract logical confusion and suggestions
            self.logical_response = get_content_between_a_b("## logical detail confusion:","## character growth confusion:", response)
            # Extract emotional confusion and suggestions
            self.emotion_response = get_content_between_a_b("## character growth confusion:","## END", response)
        else:
            warnings.warn("In reader, the generation response is empty.")
            sys.exit()

    def __call__(self)->Tuple[str, str, StoryState]:
        """
        Make the class instance callable. Triggers feedback generation and parsing.

        :return: (Tuple) Contains logical feedback (str), emotional feedback (str), and updated StoryState.
        """
        print(f"Reader is reading...")
        self.response_parser()
        return self.logical_response, self.emotion_response,self.state