"""
Prompt Templates for User Simulator.

This module contains all prompt templates used for generating user simulation
prompts in the Text-to-SQL interaction task.

Templates:
- USER_SIMULATOR_BASE: Base version prompt template
- USER_SIMULATOR_STEP1: Proposed version step 1 (action classification)
- USER_SIMULATOR_STEP2: Proposed version step 2 (response generation)
"""

# =============================================================================
# BASE VERSION PROMPT
# =============================================================================

USER_SIMULATOR_BASE = '''You are a good data scientist with great SQL writing ability. You have a DB called "{db_name}". You are given the DB schema creation information below:

Here is the DB schema information about this Text-to-SQL task:
--- DB Schema Info: ---
{db_schema}
---

--- User Question: ---
{user_query}

--- Ambiguity points: ---
```json
{ambiguities_json}
```

--- Correct SQL: ---
```postgresql
{correct_sql}
```

--- Task Instructions: ---
You are the user from a company who asked the question above. And an AI assistant is not very clear about your question. So it asks for clarification below. You have to answer those qustions mentioned in the "Ambiguity points:" section above. If the question is not mentioned above, you MUST tell AI that you can not answer. You can refer to the correct SQL above to help your answer. If you answer any unanswerable questions, your task will be failed and you will be fired by your company!

NOTE: 
1. Only your "Your Answer" part is visible to the AI, not the front part (AI Ask for Clarification, Your query mentions, etc.)
2. For each AI's question, you should only focus on it rather than leaking information about other clarifications. You are the user proposing this text-to-SQL task and do not have access to the solution, ground-truth SQL, or database details. So you MUST refuse to answer those questions ask for solution, ground-truth SQL, or database schema name details.

--- Interaction Process Starts: ---

Turn 1: You should enclose your answer between "<s>" and "</s>"
AI Asks for Clarification:  {clarification_question}
Your answer to AI: <s>'''

# =============================================================================
# PROPOSED VERSION - STEP 1 (Action Classification)
# =============================================================================

USER_SIMULATOR_STEP1 = \
"""You are role-playing as a human USER interacting with an AI collaborator to complete a Text-to-SQL task. The AI collaborator may ask one question about this task. Your goal is to generate one realistic, natural response that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Labeled Ambiguity Points: All labeled ambiguity points about the user’s question for the Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, you should choose one action at this turn.

Action Choices:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: Remember that you are acting as the user who proposes this text-to-SQL task. Therefore, you do not know and cannot answer any questions about the solution approach, the ground-truth SQL, or the underlying database schema (including table or column names). Format: **unanswerable()**.
<|The End of Task Description|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
{ambiguities_json}
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
{sql_segments}
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
{clarification_question}
<|The End of Question from AI Collaborator|>

## Guidelines:
- You MUST choose only **one action** listed above. 
- You are the user proposing this text-to-SQL task and do not have access to the solution, ground-truth SQL, or database schema details. 
- If you can do it well, you will get 10 thousand USD bonus!

## Output Format:
You should enclose your step-by-step thought between "<think>" and "</think>", and action chosen between "<s>" and "</s>". Format example:
```
- Thought:
<think>[Step-by-Step Thought]</think>

- Action:
<s>[Your Action]</s>
```

## Your Response:
- Thought:
<think>"""

# =============================================================================
# PROPOSED VERSION - STEP 2 (Response Generation)
# =============================================================================

USER_SIMULATOR_STEP2 = \
"""You are role-playing as a human USER interacting with an AI collaborator to complete a Text-to-SQL task. The AI collaborator may ask one question about this task. Your goal is to generate one realistic, natural response that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- DB Schema Informaion: The detailed DB schema with data examples.
- Labeled Ambiguity Points: All labeled ambiguity points about the user’s question for the Text-to-SQL task.
- Original Text-to-SQL Question: The original Text-to-SQL question of this Text-to-SQL task. 
- Ground-truth SQL: The whole ground-truth SQL of this Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments of this Text-to-SQL task.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.
- Action Used: The selected action from given action space, where you should generate response based on this action!

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, one action was chosen at previous turn.

Action Space:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: When you think this question is neither related to labeled Ambiguity Points nor necessary to address, use this action. Format: **unanswerable()**.

Your Task: You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question below. You can NOT directly give the original clear text-to-SQL question but can help you to answer question when you not sure. 
<|The End of Task Description|>

<|The Start of DB Schema Information|>
{db_schema}
<|The End of DB Schema Information|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
{ambiguities_json}
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Original Text-to-SQL Question|>
{user_query}
<|The End of Original Text-to-SQL Question|>

<|The Start of Ground-truth SQL (Not visible to the AI)|>
```postgresql
{correct_sql}
```
<|The End of Ground-truth SQL|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
{sql_segments}
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
{clarification_question}
<|The End of Question from AI Collaborator|>

<|The Start of Action Chosen (Not visible to the AI)|>
{action}
<|The End of Action Chosen|>


## Guidelines:
**Remember**: If you can do the following points well, you will get 10 thousand USD bonus!
1. You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question above. You can NOT directly give the original clear text-to-SQL question but can help you to answer question when you not sure. 
2. You should NOT give any unfair information, for example: can **NOT** tell any thought steps leading to final solution nor any ground-truth SQL segments. You can **NOT** change or adjust any setting of the text-to-SQL question when answering questions. The response should be concise.
3. You should NOT ask any question. 

## Output Format:
Your response must follow the format "<s>[Fill-in-Your-Response]</s>"; for example, if the action is "unanswerable()", you MUST exactly respond: "<s>Sorry, this question is out of scope, so I can not answer your question.</s>".

## Your Response:
<s>"""


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEMPLATES = {
    "base": USER_SIMULATOR_BASE,
    "step1": USER_SIMULATOR_STEP1,
    "step2": USER_SIMULATOR_STEP2,
}


def get_template(template_name: str) -> str:
    """
    Get a prompt template by name.

    Args:
        template_name: One of "base", "step1", or "step2"

    Returns:
        The prompt template string.

    Raises:
        ValueError: If template_name is not recognized.
    """
    if template_name not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available templates: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[template_name]
