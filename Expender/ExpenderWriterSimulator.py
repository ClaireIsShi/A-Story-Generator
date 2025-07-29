'''
-- @Time    : 2025/7/11 16:03
-- @File    : ExpenderWriterSimulator.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

import os
import warnings
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

from utils import get_content_between_a_b
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import set_env
from StoryState import StoryState

# Set environment variables
set_env()

EXPENDER_SYS_PRMPT = """
You're a talented story writer and a native speaker of {language}. Your task is to edit a part of the story in {language} based on the following OUTLINE:{last_outline}. Remember this: it's ok to generate or delete some details that the original outline doesn't tell, such as characters' names, emotions, logics, and personal stories, as long as they're logically appropriate, and keep as specific as possible.
"""
HUMAN_INITIAL_PROMPT = """
Now, I'm writing a story based on the following information:
topic: {topic}, Main character: {main_character}, Main Goal:{main_goal} language: {language}.
Your task is to expand specific writing based on the OUTLINE:{last_outline}, your expanded story should still be focused on this topic: {topic}. 
Follow these steps:
1. Expand the writing based on the original outline at least to {length} words;
2. It's ok to generate some details that the original outline doesn't tell, such as characters' names and personal stories, as long as they're logically appropriate and as specific as possible.
3. Add some details to make this part of the story more readable.
4. Don't overwrite the settings or information of the main characters. Make sure your story is fresh to readers who have already read the former storylines.
5. Your output should be in {language}, and your story should still be focused on this topic: {topic} and OUTLINE:{last_outline}.
Output your result without any explanation.
"""

HUMAN_REWRITE_PROMPT = """
After reading your expanded story, I find there are some logical details and character growth issues in your story. Here's my logical detail suggestion:{logical_confusion_and_suggestion}.
And here's my character suggestion:{character_growth_confusion_and_suggestion}.
Edit your last output to generate a better one that's based on my logical suggestion and character growth suggestion. Still, make sure your story is across to this topic: {topic} and OUTLINE:{last_outline}.
Follow these steps:
1. Look at the logical and character growth suggestion, and edit your last output to generate a better one that's based on my logical suggestion and emotional suggestion. Still, make sure your story is across to this topic: {topic} and OUTLINE:{last_outline}.
2. Don't overwrite the settings or information of the main characters. Make sure your story is fresh to readers who have already read the former storylines.
Output your result without any explanation. 
"""
CHANGE_OUTLINE_PROMPT = """
You're a good story writer and a native speaker of {language}. Now you get one part of your story:{story}.
The story is generated based on this outline: {last_outline} 
Your job is to see if there's any extra detail and added information in this part of the story compared to its original outline. If there is any, you should generate a new outline based on this part of the story. Output your result without any explanation.
Follow this format:
## new_outline:
<your new outline>
## END
"""


def get_new_outline(story:str):
    """
    Get the new outline from the story.
    :return: (str) The new outline.
    """
    return get_content_between_a_b ( '## new_outline:', '## END', story )

def get_whole_story(story:str):
    """
    Get the whole story from the story.
    :return: (str) The whole story.
    """
    return get_content_between_a_b ( '## whole story:', '## END', story )
from settings import WRITE_LLM
class ExpenderWriterSimulator:
    def __init__(self,state:StoryState,llm = WRITE_LLM, length:int = 800):
        """
        Initialize an instance of the Expender class.

        :param state: (StoryState) A state object containing story information, such as topic, main character, main goal, language, and the latest story outline.
        :param llm: (ChatAnthropic) A language model instance, defaulting to ChatAnthropic with a specific model.
        :param length: (int) The minimum length of the expanded story, defaulting to 800.
        """
        self.state = state
        self.llm = llm
        self.topic = self.state['Topic']
        self.main_character = self.state['MainCharacter']
        self.main_goal = self.state['MainGoal']
        self.language = self.state['Language']
        self.last_outline = self.state['RecentStory'][1]
        if self.state['StartSign']:
            self.first_line = self.state['RecentStory'][0]
        else:
            self.first_line = None
        self.length = length
        self.text = ''

    def __call__(self, logical_confusion_and_suggestion: Optional[str] = None, character_growth_confusion_and_suggestion: Optional[str] = None)->str:
        """
        Make the class instance callable. Depending on the input parameters,
        either run the initial story expansion task or rewrite the story based on suggestions.

        :param logical_confusion_and_suggestion: (str, optional) Logical issues and suggestions for rewriting the story.
        :param character_growth_confusion_and_suggestion: (str, optional) Character growth issues and suggestions for rewriting the story.
        :return: (str) The text of the expanded or rewritten story.
        """
        if logical_confusion_and_suggestion is None and character_growth_confusion_and_suggestion is None:
            print(f"Expending story...")
            # No data is passed in, run the initial task
            if self.state['StartSign']:
                self.initial_first_outline()
                return self.text
            else:
                return self.initial_last_task()

        else:
            print(f"Editing story with reviewing...")
            # Two strings are passed in, run the rewrite and update process
            return self.rewrite_and_update(logical_confusion_and_suggestion, character_growth_confusion_and_suggestion)


    def set_init_prompt(self)-> ChatPromptTemplate:
        """
        Set the initial prompt template, including system prompts and user initial input prompts.

        :return: (ChatPromptTemplate) A formatted initial chat prompt template.
        """
        system_template = EXPENDER_SYS_PRMPT
        # Create a system message prompt template
        system_message_prompt = SystemMessagePromptTemplate.from_template ( system_template )
        human_template = HUMAN_INITIAL_PROMPT
        # Create a user message prompt template
        human_message_prompt = HumanMessagePromptTemplate.from_template ( human_template )
        self.messages = [
                # Set the system prompt, describing the writer's role and task
                system_message_prompt,
                # Initial user input prompt
                human_message_prompt
            ]
        # Create a chat prompt template from the message list
        self.sys_prompt = ChatPromptTemplate.from_messages (
            messages = self.messages,
        )
        # to run formatted_messages

    def initial_last_task(self) ->str:
        """
        Execute the initial story expansion task. Invoke the language model to generate an expanded story until the story length meets the requirement.

        :return: (str) The text of the expanded story.
        """
        self.set_init_prompt()
        msg = self.sys_prompt.format_messages (
            topic=self.topic ,
            main_character=self.main_character ,
            main_goal=self.main_goal ,
            language=self.language ,
            last_outline=self.last_outline ,
            length=self.length
        )
        # Retry flag to control the loop
        button = True
        trying = 0
        while button and trying < 4:
            try:
                # Invoke the language model to generate an expanded story based on the initial prompt
                text = self.llm.invoke(msg).content
                self.text = text
                # Add the AI's response to the message list
                self.messages.append(AIMessagePromptTemplate.from_template ( self.text ))
                # Assert whether the length of the generated story meets the minimum length requirement
                assert len (self.text ) >= self.length, "The length of the expended story is less than the required length" + str(self.length) + " generation retrying..."
                # If the length meets the requirement, exit the loop
                button = False
                # Print the success message, including the original outline and the length of the expanded story

            except:
                # If an error occurs, issue a warning and continue retrying
                warnings.warn ( "Error in expending story, retrying..." )
                trying += 1
                continue
        if button:
            if len(self.last_outline)>51:
                warnings.warn ( f"Error in expending story for outline{self.last_outline[:50]}...(etc.) please try later, or change to other LLMs." )
            else:
                warnings.warn ( f"Error in expending story for outline{self.last_outline} please try later, or change to other LLMs." )

            return None
        llm = self.llm
        trying = 0
        button = True
        while button and trying < 4:
            try:
                new_outline = llm.invoke(CHANGE_OUTLINE_PROMPT.format(
                    topic = self.topic,
                    main_character = self.main_character,
                    main_goal = self.main_goal,
                    language = self.language,
                    last_outline = self.last_outline,
                    length = self.length,
                    story = self.text
                )).content
                new_outline = get_new_outline(new_outline)
                self.state["RecentStory"][-1] = new_outline
                self.last_outline = new_outline
                button = False
            except:
                trying += 1
                continue
        return self.text

    def initial_first_outline(self) -> str:
        #if self.state['StartSign']:
        self.set_init_prompt()
        # Format the prompt template and fill in specific story information
        msg = self.sys_prompt.format_messages (
                topic=self.topic ,
                main_character=self.main_character ,
                main_goal=self.main_goal ,
                language=self.language ,
                last_outline=self.first_line,
                length=self.length
            )
        button = True
        trying = 0
        while button and trying < 4:
            try:
                # Invoke the language model to generate an expanded story based on the initial prompt
                text = self.llm.invoke ( msg ).content
                # Add the AI's response to the message list

                # Assert whether the length of the generated story meets the minimum length requirement
                assert len (
                        text ) >= self.length , "The length of the expended story is less than the required length" + str (
                        self.length ) + " generation retrying..."
                # If the length meets the requirement, exit the loop
                button = False
                # Print the success message, including the original outline and the length of the expanded story
                self.messages.append(AIMessagePromptTemplate.from_template ( text ))
            except:
                # If an error occurs, issue a warning and continue retrying
                warnings.warn ( "Error in expending story, retrying..." )
                trying += 1
                continue
        if button:
            warnings.warn ( f"Error in expending story for outline{self.first_line[:50]} please try later, or change to other LLMs." )
            return None

        self.text = text
        trying = 0
        button = True
        while button and trying < 4:
            try:
                new_outline = self.llm.invoke ( CHANGE_OUTLINE_PROMPT.format (
                        topic=self.topic ,
                        main_character=self.main_character ,
                        main_goal=self.main_goal ,
                        language=self.language ,
                        last_outline=self.first_line ,
                        length=self.length ,
                        story=self.text
                    ) ).content
                new_outline = get_new_outline ( new_outline )
                self.state["RecentStory"][0] = new_outline
                button = False
            except:
                trying += 1
                continue
        return text

    def set_startsign_to_false(self):
        self.state['StartSign'] = False


    def rewrite(self, logical_confusion_and_suggestion:str, character_growth_confusion_and_suggestion:str):
        """
        Rewrite the story based on logical and character_growth suggestions.

        :param logical_confusion_and_suggestion: (str) Logical issues and suggestions.
        :param character_growth_confusion_and_suggestion: (str) character_growth issues and suggestions.
        :return: (str) The text of the rewritten story.
        """
        # Format the user rewrite prompt template
        human_template = HUMAN_REWRITE_PROMPT
        # Add the user rewrite prompt to the message list
        self.messages.append(HumanMessagePromptTemplate.from_template ( human_template ))
        ####################################################################
        # 此时self.messages:[sys, human_init, AI, human反馈]
        ####################################################################
        # Invoke the language model to generate a rewritten story based on the message list
        if self.state['StartSign']:
            formatted_messages = ChatPromptTemplate.from_messages ( self.messages ).format_messages (
                topic=self.topic ,
                main_character=self.main_character ,
                main_goal=self.main_goal ,
                language=self.language ,
                last_outline=self.first_line ,
                length=self.length ,
                logical_confusion_and_suggestion=logical_confusion_and_suggestion ,
                character_growth_confusion_and_suggestion=character_growth_confusion_and_suggestion
            )

        else:
            formatted_messages = ChatPromptTemplate.from_messages (self.messages).format_messages(
            topic = self.topic,
            main_character = self.main_character,
            main_goal = self.main_goal,
            language = self.language,
            last_outline = self.last_outline,
            length = self.length,
            logical_confusion_and_suggestion = logical_confusion_and_suggestion,
            character_growth_confusion_and_suggestion = character_growth_confusion_and_suggestion
        )
        self.text = self.llm.invoke(formatted_messages).content



    def update_msg_list(self):
        """
        Update the message list after the story has been rewritten.
        Remove the last two elements from the message list and add the AI's rewritten response.

        :return: (str) The text of the rewritten story.
        """
        # The original code here has an error. The pop method cannot accept a list parameter. It should be removed one by one.
        # Remove the last two elements from the message list.
        # This is typically done to clean up the temporary user rewrite prompt and the previous interaction.
        system_template = EXPENDER_SYS_PRMPT
        # Create a system message prompt template
        system_message_prompt = SystemMessagePromptTemplate.from_template ( system_template )
        human_template = HUMAN_INITIAL_PROMPT
        # Create a user message prompt template
        human_message_prompt = HumanMessagePromptTemplate.from_template ( human_template )
        AI_message_prompt = AIMessagePromptTemplate.from_template ( self.text )
        self.messages = [
                # Set the system prompt, describing the writer's role and task
                system_message_prompt,
                # Initial user input prompt
                human_message_prompt,
                AI_message_prompt
            ]


    def rewrite_and_update(self,logical_confusion_and_suggestion:str, character_growth_confusion_and_suggestion:str)->str:
        """
        Rewrite the story based on logical and emotional suggestions and then update the message list.
        This method first checks if the message list is in the expected format, then triggers the story rewrite process.
        After the rewrite, it updates the message list to maintain the correct conversation history.
        Finally, it verifies that the message list is back in the expected format and returns the rewritten story text.

        :param logical_confusion_and_suggestion: (str) Logical issues and suggestions for rewriting the story.
        :param character_growth_confusion_and_suggestion: (str) Character growth issues and suggestions for rewriting the story.
        :return: (str) The text of the rewritten story.
        """
        self.update_msg_list()
        # Ensure that the message list is in the correct format before rewriting.
        # Expected format: [system prompt, initial human prompt, AI response]
        assert len(self.messages) == 3, "The message list is not in the correct format. It should be [sys_prompt, human_init_prompt, AI_response]."
        # Call the rewrite method to rewrite the story based on the provided suggestions.
        self.rewrite(logical_confusion_and_suggestion, character_growth_confusion_and_suggestion)
        # Update the message list after the story has been rewritten.
        # Remove the temporary user feedback and add the new AI response.
        self.update_msg_list()
        # Ensure that the message list is back in the correct format after the update.
        # Expected format: [system prompt, initial human prompt, new AI response]
        assert len (self.messages ) == 3 , "The message list is not in the correct format. It should be [sys_prompt, human_init_prompt, AI_response]."
        # Return the text of the rewritten story.
        return self.text