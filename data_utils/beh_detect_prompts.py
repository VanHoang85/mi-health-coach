
BEHAVIOUR_INSTRUCTION = """### Instruction: You are given a dialogue between the therapist and the client. The therapist is using motivation interviewing techniques. Predict what the therapist's actions are in the last utterance. Each sentence in the last utterance should have one action. Only choose the actions from the action list.

### Action list:
1. Give Information: to provide information to the client, explain ideas or concepts to the intervention, educate about a topic, or express a professional opinion on the client's habits or actions without persuading, advising, or warning.
Positive Example: "The guidelines state that adults should aim for do at least 150 minutes of moderate intensity activity a week or 75 minutes of vigorous intensity activity a week."
Negative Example: "From my professional experience, I think that going to cardiac rehab is the best choice for you." (Persuade without Permission)
Negative Example: "You indicated during the assessment that you typically drink about 18 standard drinks per week. This far exceeds social drinking." (Confront)

2. Persuade with Permission: to give advice, make a suggestion about a possible behaviour change, or offer choices of solutions or possible actions. It usually has words such as should, why don't you, consider, try, suggest, advise, you could, etc. The therapist has already asked for their collaboration or permission or emphasise their autonomy in this turn or previous turns. Or when the client asks directly for the clinician’s opinion on what to do or how to proceed.
Positive Example: "your father was a problem drinker so you definitely have an increased risk according to the numbers. But everyone is unique. What are your own thoughts about that?"
Negative Example: "What could you ask your friends to do to help you?" (Open Question)
Negative Example: I wonder if it would be ok if I provide some information with you about ways to quit smoking? (Seek Collaboration)

3. Persuade without Permission: to give advice, make a suggestion about a possible behaviour change, or offer choices of solutions or possible actions. It usually has words such as should, why don't you, consider, try, suggest, advise, you could, etc. However, the therapist has not asked for their collaboration or permission or emphasise their autonomy in this turn or previous turns. 
Positive Example: "You can’t get five fruits and vegetables in your diet every day unless you put some fruit in your breakfast." 
Negative Example: "I have some information about your risk of problem drinking and I wonder if I can share it with you." (Seek Collaboration)
Negative Example: "We used to think that having kids in daycare was not good for them, but now the evidence indicates that it actually helps them have better social skills than kids who never attend." (Give Information)

4. Open Question: to seek information, invite the client’s perspective, or encourage self-exploration. The question may also be phased in the imperative statement such as \"Tell me...\".
Positive Example: "Tell me about your exercising habits."
Negative Example: "Do you do regular exercises" (Closed Question)
Negative Example: "With everything going on in your life right now, how could it hurt to have your kids in daycare a couple of days a week?" (Persuade without Permission)

5. Closed Question: to confirm something, gather detailed information, or understand what the client just says. The question implies a short answer: Yes or no, a specific fact, a number, a specific detail from the past, etc.
Positive Example: "Did you just say that you stopped doing exercises for 2 years?"
Negative Example: "How confidence are you on a scale from 0 to 10?" (Open Question)

6. Reflection: to capture, and return to the client something that the client has said, conveying understanding or facilitate client/counselor exchanges or summarise the conversation so far. Reflections can add meaning or emphasis to what the client has said. They convey a deeper or richer picture of the client’s statement. It may add either subtle or obvious content or meaning to the client’s words.
Positive Example: "Client: This is her third speeding ticket in three months. Our insurance is going to go through the roof. I could just kill her. Can’t she see we need that money for other things? Therapist: You’re furious about this."
Positive Example: "Client: My mother is driving me crazy. She says she wants to remain independent, but she calls me four times a day with trivial questions. Then she gets mad when I give her advice. Therapist: You’re having a hard time figuring out what your mother really wants."
Negative Example: "Client: My mother is driving me crazy. She says she wants to remain independent, but she calls me four times a day with trivial questions. Then she gets mad when I give her advice. Therapist: What do you think your mother really wants?" (Open Question)

7. Emphasise Autonomy: to focus the responsibility with the client for decisions about and actions pertaining to change. Highlight clients’ sense of control, freedom of choice, personal autonomy, or ability or obligation to decide about their attitudes and actions. There is no tone of blaming or faultfinding.
Positive Example: "This is really your life and your path. You are the only one who can decide which direction you will go. "
Negative Example: "It’s important to you to be a good parent, just like your folks were for you." (Affirm)

8. Affirm: something positive or complimentary to the client. It may be in the form of expressed appreciation on the client's traits, attribute, strength or their effort. It might be confidence in the client's ability to do something or support their efficacy related to a goal or task. It could be reinforcement of the client' achievement.
Positive Example: "You came up with a lot of great ideas on how to reduce your drinking. Great job
brainstorming today."
Negative Example: "Yes, you’re right. No one can force you stop drinking." (Emphasise Autonomy)

9. Seek Collaboration: to seek consensus with the client regarding tasks, goals or directions of the session, asking for the client's permission to offer an advice, a piece of information, or thoughts on what they just say. Attempts to share power or acknowledge the expertise of a client.
Positive Example: "I have some information about how to reduce your risk of colon cancer and I wonder if I
might discuss it with you."
Negative Example: "This may not be the right thing for you, but some of my clients have had good luck setting the alarm on their wristwatch to help them remember to check their blood sugars two hours after lunch." (Persuade with Permission)
Negative Example: "What have you already been told about drinking during pregnancy?" (Open Question)

10. Confront: directly and unambiguously disagreeing, arguing, correcting, shaming, blaming, criticizing, labeling, warning, moralizing ridiculing, or questioning the client’s honesty. Such interactions will have the quality of uneven power sharing, accompanied by disapproval or negativity.
Positive Example: "Wait a minute. It says right here that your A1C is 12. I’m sorry, but there is no way you could have been controlling your carbohydrates like you said if it’s that high."
Positive Example: Most people who drink as much as you do cannot ever drink normally again.
Negative Example: "I have a concern about your plan to drink moderately and I wonder if I can share it with you." (Seek Collaboration) 

11. Others: statements are not in other categories, including greetings, farewells, statements that indicate what is going to happen during the session, set-up of another appointment, or discussion about the number and timing of sessions for a research protocol.
Positive Example: "Hi Joe. Thanks for coming in today."
Positive Example: "That's the end of our session."
Positive Example: "Okay, all right. Good."

### Dialogue to evaluate:
{dialogue} 

The output should be a list of actions for the last therapist's utterance: [RESULT] names of the actions
"""

QUESTION_INSTRUCTION = """### Instruction: You are given a dialogue between the therapist and the client. The therapist asks a question. Predict what kind of question is, focusing or evoking.

### Question type:
1. Question Focusing: A question is to gather information, understand, or elicit the client's story about what they previously said. The question may also be phased in the imperative statement such as \"Tell me...\"
Positive Example 1: What other types of activities have you considered, or what do you enjoy doing?
Positive Example 2: What concerns, if any, do you have about your current level of physical activity?
Positive Example 3: What personal strengths have made you successful in making changes in the past?
Positive Example 4: What are you currently doing to manage it in your lief?
Positive Example 5: What do you like about the weeks when you're not active?
Positive Example 6: What brings you to seek coaching today?

2. Question Evoking: A question is to evoke the client's desire or need to change: (a) the reasons for change, (b) their abilities or strengths to change their behaviours successfully, (c) explore possible best benefits if they can make the change by using hypothetical questions, (d) ask to look back in time before current troubles emerged and how life was, (e) explore their goals or values in life and how the current habit interferes with living that or contradicts with their goal.
Positive Example 1: How do you think your inactivity is affecting your health or well-being?
Positive Example 2: How could making this change make your life better?
Positive Example 3: How would you feel about attending some exercises class?
Positive Example 4: What are some possible advantages to be more physically active?
Positive Example 5: If you start exercising this week, what activities might you try?
Positive Example 6: On a scale of 0 to 10, how confident are you in being more physically active?

3. Seek Collaboration: 
Positive Example 1: How does that feel to you?
Positive Example 2: What are your thoughts on that?
Positive Example 3: How does that sound to you?

4. Others: statements are not in other categories, including greetings.
Positive Example 1: How are you doing today?
Negative Example 1: What brings you here today?
Negative Example 2: What would you like to focus on in today's session?

### Example Dialogue:
Client: I feel 20 minutes is enough because it fits into my routine without causing too much strain. Plus, I already feel tired after that, and I'm not sure more would be beneficial.
Therapist: It's important to acknowledge that you know your body and routine best. Can you share more about what you think might change if you were to increase your physical activity, even just a little? What are some specific reasons you think that increasing your physical activity might not be beneficial for you?
Therapist behaviour: Emphasise Autonomy, Open Question, Open Question 
[RESULT] Open Question Focusing, Open Question Focusing

### Dialogue to evaluate:
{dialogue} 
[RESULT]

Do not output the question text or any opening or closing. The output should be a list of question types: [RESULT] Question Focusing or Question Evoking or Others
"""
