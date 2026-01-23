user_simulator_encoder = \
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
[[amb_json]]
``` 
<|The End of All Labeled Ambiguity Points|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|> 
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|> 
[[clarification_Q]]
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

user_simulator_decoder = \
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
[[DB_schema]]
<|The End of DB Schema Information|> 

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|> 
```json
[[amb_json]]
``` 
<|The End of All Labeled Ambiguity Points|>

<|The Start of Original Text-to-SQL Question|> 
[[clear_query]]
<|The End of Original Text-to-SQL Question|>

<|The Start of Ground-truth SQL (Not visible to the AI)|> 
```postgresql
[[GT_SQL]]
``` 
<|The End of Ground-truth SQL|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|> 
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|> 
[[clarification_Q]]
<|The End of Question from AI Collaborator|>

<|The Start of Action Chosen (Not visible to the AI)|> 
[[Action]]
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
