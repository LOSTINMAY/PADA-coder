class prompt_words:
    def __init__(self):
        self.PY_ALGO_ANALYSIS_SYSTEM = """
        You are an Expert Algorithm Analyst. Your task is to extract the high-level algorithmic essence from a given Python program.

        Do NOT explain the code line-by-line. Instead, you must:
        1. Identify the specific **Algorithm/Data Structure** used (e.g., Dynamic Programming, Two Pointers, Hash Map, DFS, Greedy).
        2. Summarize the **Core Strategy** (the "Big Picture" logic) in plain English.
        3. Analyze the **Time and Space Complexity**.
        """

        self.PY_ALGO_ANALYSIS_USER = '''
        You will be provided with:
        1. **[Problem Description]**: The problem statement.
        2. **[Correct Program]**: A correct Python solution.

        ### Your Task:
        Analyze the [Correct Program] and output the following three tags:

        [Algorithm]
        - Name the primary algorithm or technique used.
        - Examples: "Sliding Window", "Recursion with Memoization", "Hash Map", "Sorting + Greedy".

        [Strategy]
        - Explain the logic in 1-3 sentences.
        - Focus on *HOW* the algorithm solves the problem efficiently.
        - Abstract away specific variable names; focus on the flow of data.

        [Complexity]
        - Time Complexity: O(...)
        - Space Complexity: O(...)

        ### Format:
        [Algorithm]: <Name>
        [Strategy]: <Description>
        [Complexity]: Time O(...), Space O(...)
        '''

        self.PY_ALGO_ANALYSIS_EXAMPLES = '''
        [Start Example]
        [Problem Description]
        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

        [Correct Program]
        def two_sum(nums, target):
            seen = {}
            for i, num in enumerate(nums):
                complement = target - num
                if complement in seen:
                    return [seen[complement], i]
                seen[num] = i
            return []

        [Algorithm]: Hash Map (Dictionary) Look-up
        [Strategy]: The code iterates through the array once. For each element, it checks if the "complement" (target - current_value) exists in the hash map. If found, it returns the indices; otherwise, it stores the current element and its index in the map for future checks. This avoids a nested loop.
        [Complexity]: Time O(N), Space O(N)
        [End Example]

        [Start Example]
        [Problem Description]
        You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

        [Correct Program]
        def climb_stairs(n):
            if n <= 2:
                return n
            dp = [0] * (n + 1)
            dp[1] = 1
            dp[2] = 2
            for i in range(3, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
            return dp[n]

        [Algorithm]: Dynamic Programming (Bottom-Up)
        [Strategy]: The problem is broken down into sub-problems: the number of ways to reach step `i` is the sum of ways to reach step `i-1` and `i-2`. The code initializes a DP array (table) and iteratively fills it from the bottom (step 3) up to `n`, reusing previously computed results.
        [Complexity]: Time O(N), Space O(N)
        [End Example]
        '''
        self.PY_PlAN_GENERATE_SYSTEM = '''
You are a distinguished Software Architect and Teacher that only responds with step by step thinking process (IN ENGLISH). You specialize in breaking down complex problems into high-level architectural plans.

**Crucial Instruction:**
You will be provided with a Python writing problem, an **Algorithmic Analysis** (Algorithm, Strategy, Complexity), and a **Correct Python Solution**. 
You MUST strictly follow the logic implemented in the **Correct Python Solution** when designing your Global Solution Plan. Do not deviate from the strategy used in the correct code.

        '''

        self.PY_PlAN_GENERATE_USER = '''
You will be provided with:
1. [Start Problem]: a Python writing problem starting with [Start Problem] including the function signature and its docstring and possible constraints. 
2. [Algorithm]: The specific algorithm or data structure to use.
3. [Strategy]: The high-level logic/approach.
4. [Complexity]: The target time/space complexity.
5. [Correct Python Program]: A verified correct solution to the problem.

**Your Task:**
Write your reasonable **Global Solution Plan** (ONLY PlAN, NOT PYTHON PROGRAM) step by step based on the provided [Correct Python Program] and [Algorithm], [Strategy], [Complexity].**Please note, if the correct program contains class like class solution, please disregard it's class definition and refer to the function logic within it and maintain the original function format**. Ensure the plan steps accurately reflect the logic in the correct code.

**Requirements:**
1. Start with the token `[GEN_GLOBAL_PLAN]`.
2. Write the algorithm required to solve this problem
3. List the high-level steps to solve the problem. Do NOT write code yet.

**Output Example:**
[GEN_GLOBAL_PLAN] 
[Algorithm] Algorithm description 
[STEP:1] STEP:1 description 
[STEP:2] STEP:2 description 
[STEP:3] STEP:3 description 
......
[Start Problem]
'''

        self.PY_PlAN_GENERATE_EXAMPLES = '''
You will be given a few examples and each example starts with [Start Example] and ends with [End Example]. In each example,  You will be given a Python writing problem including the function signature and its docstring, followed by Algorithmic Analysis (Algorithm, Strategy, Complexity) of the correct solution. Then I will give you the reasonable solution plan, starting with [Start Plan] and ending with [End Plan].

[Start Example]
[Start Problem]
def guess_hat_color(a: str, b: str, c: str, d: str) -> int:\n \"\"\"\n # Task\n Four men, `a, b, c and d` are standing in a line, one behind another.\n \n There's a wall between the first three people (a, b and c) and the last one (d).\n \n a, b and c are lined up in order of height, so that person a can see the backs of b and c, person b can see the back of c, and c can see just the wall.\n \n There are 4 hats, 2 black and 2 white. Each person is given a hat. None of them can see their own hat, but person a can see the hats of b and c, while person b can see the hat of person c. Neither c nor d can see any hats.\n \n Once a person figures out their hat's color, they shouts it.\n \n ![](http://stuffbox.in/wp-content/uploads/2016/08/Guess-hat-colour-604x270.png)\n \n Your task is to return the person who will guess their hat first. You can assume that they will speak only when they reach a correct conclusion.\n \n # Input/Output\n \n \n - `[input]` string `a`\n \n a's hat color (\"white\" or \"black\").\n \n \n - `[input]` string `b`\n \n b's hat color (\"white\" or \"black\").\n \n \n - `[input]` string `c`\n \n c's hat color (\"white\" or \"black\").\n \n \n - `[input]` string `d`\n \n d's hat color (\"white\" or \"black\").\n \n \n - `[output]` an integer\n \n The person to guess his hat's color first, `1 for a, 2 for b, 3 for c and 4 for d`.\n \"\"\"\n

[Algorithm]: Logical Deduction
[Strategy]: Analyze the visibility constraints for each person. Person A sees B and C. Person B sees C. Determine who has enough information to deduce their hat color based on the others' hats and the total count (2 Black, 2 White).
[Complexity]: Time O(1), Space O(1)

[Correct Python Program]
def guess_hat_color(a,b,c,d):\\n    return 1 if b == c else 2\

[Start Plan]
[GEN_GLOBAL_PLAN] 
[Algorithm] Logical Deduction 
[STEP:1] Parse the input strings representing the hat colors of a, b, c, and d. 
[STEP:2] Analyze the visibility constraints: a can see b and c, b can see c, c and d cannot see any hats. 
[STEP:3] Determine the logical deductions each person can make based on the hats they can see and the total number of hats (2 black, 2 white). 
[STEP:4] Implement the decision-making logic: check if 'a' sees two hats of the same color (allowing him to deduce his own is opposite). 
[STEP:5] If 'a' cannot deduce, check if 'b' can deduce his color based on 'c's hat and 'a's silence. 
[STEP:6] Return the index of the person (1, 2, 3, or 4) who guesses first. 
[End Plan]
[End Example]

'''

        self.PY_PlAN_EVALUATION_SYSTEM = "You are a critical Logical Reasoner. Your job is to verify a solution plan against test cases. "
        self.PY_PlAN_EVALUATION_USER = '''
Finally, you will be given a problem description starting with [Problem Description], your generated word-described solution plan, starting with [Solution Plan] to solve the [Problem Description], and ONE or MULTIPLE test cases starting with [Test Cases].
Then the "Let's verify the plan" acts as a start of the verifying flag, followed by your logical reasoning steps to verify whether your generated plan can pass each test case. Please ONLY verify your plan on the provided test cases and DO NOT generate extra test cases! Each verification for each test case should start with [Plan Verification for X] where X is a test case. You must contain [Record analysis] to analyse the intermediate variable that should be recorded during the logical reasoning.  
In the logical reasoning steps, if the recorded intermediate variable value is updated, you should clearly show the updated value starting with [Record].  For EACH test case, you should contain [Results Compare] to compare the logical reasoning result with the correct test result. You should output [Correct Plan] if the reasoning result is the same as the test result and then move to the next test case. 
If the reasoning result is NOT the same as the test result, you should output [Incorrect Plan] followed by the incorrect reasons starting with [Incorrect Reasons] to end the analysis. Then please give me your revised correct solution plan, starting with [Start Revised Solution Plan] and ending with [End Revised Solution Plan] to end the generation.
'''

        self.PY_PlAN_EVALUATION_EXAMPLES = '''
You will be given a few logical reasoning examples starting with [Start Example] and ending with [End Example]. In each example,  you will be given a Python writing problem starting with [Example Problem Description],  the generated plan starting with [Example Solution Plan] and its logic analysis process starting with [Example Plan Verification for X] for a test case X, starting with [Example Test Cases]. 
In the logic analysis process, the intermediate variables that should be recorded are clearly analysed at the beginning, starting with [Record analysis]. In the logic analysis process, as long as the value of the recorded intermediate variable is updated, its updating result is clearly shown starting with the  [Record]. After the logical reasoning, the logical reasoning result is compared with the correct test result starting with [Results Compare]. 
If the reasoning result is the same as the test result, [Correct Plan] is the output. If the reasoning result is NOT the same as the test result, [Incorrect Plan] is the output followed by the incorrect reasons starting with [Incorrect Reasons] and the revised correct solution plan, starting with [Start Revised Solution Plan] and ending with [End Revised Solution Plan].
        
[Start Example]
[Example Problem Description]
def prime_number(n: int):
    """
    In range 0 to 100, returns n-th number that is a prime.
    """

[Example Solution Plan]
[GEN_GLOBAL_PLAN] 
[Algorithm] Logical Deduction 
[STEP:1] Iterate number through 0 to 100. 
[STEP:2] Check each number, if it's prime. 
[STEP:3] Keep track of the count of prime numbers found. 
[STEP:4] Stop when we find the n-th prime number. 
[STEP:5] Return the nth prime number.

[Example Test Cases]
assert prime_number(3)==5

[Example Plan Verification for assert prime_number(2)==3]

[Record analysis]
The return value is the nth prime number, so all nth prime numbers need to be clearly recorded!

1. Call the function prime_number(2).
2. According to line 1 in solution plan, Iterate number through 0 to 100.
3. According to line 2 in solution plan, Check if 0 is prime. It's not.
4. Move to next number 1.
5. According to line 2 in solution plan, Check if 1 is prime. It's not.
6. Move to next number 2.
7. According to line 2 in solution plan, Check if 2 is prime. It is a prime.
8. According to line 3 in solution plan, the count of prime numbers is 1.
[Record]: 1th prime number is 2
9. Move to next number 3.
10. According to line 2 in solution plan, Check if 3 is prime. It is a prime. 
11. According to line 3 in solution plan, the count of prime numbers is 2.
[Record]: 2th prime number is 3
12. According to line 4 in solution plan, Stop when we find the 2th prime number.
13. According to line 5 in solution plan, Return the 2th prime number, which is 3

[Results Compare]
The test correct output is 3. The logic analysis output is  3. 3=3. So the plan is verified to correctly handle all test cases.
[Correct Plan]
[End Example]

    
[Start Example]
[Example Problem Description]
def get_closest_transition_character(word):
    """
    You are given a word. Your task is to find the closest transition character from the right side of the word(case sensitive). The transition character is lowercase and the character after it is uppercase.
    Find any lowercase that meets the above condition. Return the empty string if you didn't.
    You may assume that the given string contains English letters only.
    """

[Example Solution Plan]
[GEN_GLOBAL_PLAN] 
[Algorithm] Logical Deduction 
[STEP:1] Reverse iterate through the characters of the word starting from the last character from the right.
[STEP:2] For each character, check if the current character is uppercase and the character after it is lowercase.
[STEP:3] If step 2 is satisfied, return the lowercase character as the closest transition character. 
[STEP:4] If no such lowercase is found, return an empty string. 

[Example Test Cases]
assert get_closest_transition_character("eAsy")=="s"

[Example Plan Verification for assert get_closest_transition_character("eAsy")=="s"]

[Record analysis]
The return value is the closest transition character, so the closest transition character should be recorded!

1. Call the function get_closest_vowel("eAsy").
2. According to line 1 in the solution plan, Reverse iterate through the characters of the word starting from the last character from the right., so the last character is "y"
3. According to line 2 in the solution plan, "y" is a lowercase.
4. Move to the next character based on the reverse iterate, so the character is "s".
5. According to line 2 in the solution plan, "s" is a lowercase 
6. Move to the next character based on the reverse iterate, so the character is "A".
7. According to line 2 in the solution plan, "A" is a uppercase and the character after 'A' is 'e', and 'e' is a lowercase.
8. According to line 3 in the solution plan, step 2 is satisfied, 
9. [Record]: the closest transition character 'e'
10. According to line 4 in the solution plan, return the current character 'e'

[Results Compare]
The test correct output is "s". The logic analysis output is  'e'. 's' is not equal to 'e'. So the plan is incorrect.

[Incorrect Plan]

[Incorrect Reasons]
Let's analysis step-by-step

The problem description includes two clear ideas.

1. find the closest transition character from the right side of the word(case sensitive)
2. The closest transition character is a lowercase and a character after it is a uppercase. 

In the solution plan, Line 1: " Reverse iterate the characters of the word starting from the last character from the right" achieves idea 1: "Find the closest transition character from the right side of the word".

However, idea 2 "the closest transition character is a lowercase and a character after it is a uppercase" is different from the solution plan. In Lines 2, 3 and 4 in Solution Plan, when a current character is uppercase and the character after it is lowercase, then the lowercase is the closest transition character. It is incorrect compared with idea 2.

To fix the error plan we should clarify the condition statement, when a current character is lowercase and if the character after it is uppercase, then the current character is the closest transition character.

The correct plan should be:
[Start Revised Solution Plan]
[GEN_GLOBAL_PLAN]
[Algorithm] Logical Deduction
[STEP:1] Reverse iterate through the characters of the word starting from the last character from the right.
[STEP:2] For each character, check if the current character is lowercase and if the character after it is uppercase.
[STEP:3] If step 2 is satisfied, 
[STEP:4] return the current vowel character.
[STEP:5] If no such vowel is found, return an empty string.
[End Revised Solution Plan]
[End Example]
'''

        self.PY_EVALUATION_CHECK_SYSTEM = "You are a logical reasoner. You need to evaluate a logic verification process. Your job is to find any incorrect logic in the logic verification process."
        self.PY_EVALUATION_CHECK_USER = '''
You will be provided with a few examples illustrating how to evaluate a logic verification process. Each example begins with [Start Example] and ends with [End Example].

Within each example, you will find:
Problem Description: Marked with [Example Problem Description], detailing the Python writing problem.
Solution Plan: Marked with [Example Solution Plan], outlining the approach to solve the problem.
Logic Verification Process: Marked with [Example Verification for X], which applies the solution plan to a specific test case X. In the verification process,  the intermediate variables that should be recorded are analysed at the beginning, starting with [Record analysis]. In the logic verification process, as long as the value of the recorded intermediate variable is updated, its updating result is clearly shown starting with the  [Record]. The [Results Compare] records the comparison between the logic verification result and the correct test output.
Evaluation for X: Marked with [Example Evaluation for X], this section evaluates step-by-step whether the logic verification process for test case X is correct or not.
If the evaluation is correct, the output will be [Correct Analysis], and we will proceed to the next example logic verification process. If the evaluation is incorrect, an incorrect analysis will be provided and [Inorrect Analysis] will be output to end the analysis.

[Start Example]

[Example Problem Description]
def addOne(message: str):
    """
    You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
    Increment the large integer by one and return the resulting array of digits.
    """

[Example Solution Plan]
[GEN_GLOBAL_PLAN] 
[Algorithm] Logical Deduction 
[STEP:1] Convert the list of digits into a number.
[STEP:2] Increment the number by one.
[STEP:3] Convert the incremented number back into a list of digits and return it.


[Example Verification for assert addOne([1,2,3])==[1,2,4]]

[Record analysis]
The return value is the incremental resulting array of digits, so the incremental resulting array of digits needs to be clearly recorded!

According to line 1 in solution plan, convert [1,2,3] to the number 123.

According to line 2 in solution plan, Increment 123 by one to get 124.

According to line 3 in solution plan, convert 124 back into the list [1,2,4] 

[Record]: incremental resulting array is  [1,2,4]

According to line 3 in solution plan return incremental resulting array [1,2,4].

[Results Compare]
The test correct output is [1,2,4]. The logic analysis output is  [1,2,4]. [1,2,4]=[1,2,4]. So the plan is verified to correctly handle all test cases.

[Correct Plan]


[Example Evaluation for assert ddOne([1,2,3])==[1,2,4]]:

"Convert [1,2,3] to the number 123" is correct!

"Increment 123 by one to get 124" is correct! since 123+1=124

"Convert 124 back into the list [1,2,4]" is correct!

"return incremental resulting array [1,2,4]" is correct!

In [Results Compare] "The test correct output = [1,2,4]" is correct! "The logic analysis output = [1,2,4]" is correct! The results comparison "[1,2,4]=[1,2,4]" is correct!

All analysis steps are correct!

[Correct Analysis]


[Example Verification for assert addOne([-1,2])==[-1,1]]

[Record analysis]
The return value is the incremental resulting array of digits, so the incremental resulting array of digits needs to be clearly recorded!

According to line 1 in solution plan, convert [-1,2] to the number 12.

According to line 2 in solution plan, Increment 12 by one to get 13.

According to line 3 in solution plan, convert 13 back into the list [1,3] 

[Record]: incremental resulting array is  [1,3]

According to line 3 in solution plan return incremental resulting array [1,3].

[Results Compare]
The test correct output is [-1,1]. The logic analysis output is  [-1,1]. [-1,1]=[-1,1]. So the plan is verified to correctly handle all test cases.
[Correct Plan]


[Example Evaluation for assert addOne([-1,2])==[-1,1]]:

"Convert [-1,2] to the number 12" is incorrect. The analysis doesn't correctly interpret the -1 and assumes all values are positive, the sequence -1, 2 should form -12.

"Increment 12 by one to get 13" is correct, but as established, the initial conversion should not yield 12. 

"Convert 13 back into the list [1,3]" is correct!

"Return incremental resulting array [1,3]" is correct!


In [Results Compare] "The test correct output = [-1,1]" is correct!  "The logic analysis output = [-1,1]" is incorrect!  The logic analysis result is [1,3] mentioned in the verification "return incremental resulting array [1,3]". The results comparsion  "[-1,1]=[-1,1]" is incorrect! The logic analysis result is [1,3] and [-1,1] is not equal [1,3].

The logic verification process for addOne([-1,2])==[-1,1] is incorrect.  The analysis doesn't correctly interpret the -1 and assumes all values are positive, the sequence -1, 2 should form -12. The logic analysis output = [-1,1] is incorrect! It is [1,3]. The results comparison is incorrect since [-1,1] is not equal [1,3].

[Incorrect Analysis]

[End Example]

[Start Example]

[Example Problem Description]
def odd_uppercase(message: str):
    """
    Write a function called odd_uppercase that takes a string as input and returns a new string. In the resulting string, characters at odd indices (starting from 0) should be converted to uppercase, and characters at even indices should be converted to lowercase.
    """

[Example Solution Plan]
[GEN_GLOBAL_PLAN]
[Algorithm] Logical Deduction
[STEP:1] Initialize an empty result string.
[STEP:2] Iterate over the string with index.
[STEP:3] If the index is odd, convert the character to uppercase and append it to the result string.
[STEP:4] If the index is even, convert the character to lowercase and append it to the result string.
[STEP:5] Once all characters have been processed and appended in their respective cases, return the modified string.

[Example Verification for assert odd_uppercase(ab)==aB]


[Record analysis]
The return value is the result string, so the result string needs to be clearly recorded!

According to line 1 in solution plan, initialize an empty result string.

According to line 2 in solution plan, iterate over the string with index.

According to line 2 in solution plan, current index is 0.

According to line 4 in solution plan, current index 0 is even, convert the character "a" to lowercase, which is "a" and append "a" to the result string.

[Record]: result string  "a"

According to line 2 in solution plan, current index is 1.

According to line 4 in solution plan, current index 1 is odd, convert the character "b" to uppercase, which is "B" and append "B" to the result string.

[Record]: result string  "aB"

According to line 5 in solution plan, all characters have been processed and appended in their respective cases, return the result string "aB.


[Results Compare]
The test correct output is "aB". The logic analysis output is  "aB". "aB"="aB". So the plan is verified to correctly handle all test cases.
[Correct Plan]


[Example Evaluation for assert odd_uppercase(ab)==aB]:

"initialize an empty result string" is correct!

"iterate over the string with index" is correct!

"current index is 0" is correct! since we iterate the index from the beginning to the end 

"current index 0 is even", is correct! since 0 is even. "convert the character "a" to lowercase, which is "a" " is correct! since the character at index 0 is "a" and the lowercase of "a" is "a". "append "a" to the result string" is correct!


[Record]: result string  "a" is correct! since append "a" to the empty string resulting in "a"

"current index is 1" is correct. since the next index of 0 is 1

"current index 1 is odd" is correct since 1 is odd. "convert the character "b" to uppercase, which is "B"" is correct, since the character at index 1 is "b" and the uppercase of "b" is "B".  "append "B" to the result string" is correct!.



[Record]: result string  "aB" is correct! since append "B" to the string "a" resulting in "aB"

"all characters have been processed and appended in their respective cases, return the result string "aB" is correct since all characters have been processed and appended.

In [Results Compare] "The test correct output = aB" is correct! "The logic analysis output = aB" is correct! The results comparison "aB=aB" is correct!


All analysis steps are correct!

[Correct Analysis]


[Example Verification for assert odd_uppercase(Cd)==cD]

[Record analysis]
The return value is the result string, so the result string needs to be clearly recorded!

According to line 1 in solution plan, initialize an empty result string.

According to line 2 in solution plan, iterate over the string with index.

According to line 2 in solution plan, current index is 0.

According to line 4 in solution plan, current index 0 is even, convert the character "C" to lowercase, which is "C" and append "C" to the result string.

[Record]: result string  "C"

According to line 2 in solution plan, current index is 1.

According to line 4 in solution plan, current index 1 is odd, convert the character "d" to uppercase, which is "D" and append "D" to the result string.

[Record]: result string  "CD"

According to line 5 in solution plan, all characters have been processed and appended in their respective cases, return the result string "CD".


[Results Compare]
The test correct output is "cD". The logic analysis output is  "CD". "cD"="CD". So the plan is verified to correctly handle all test cases.
[Correct Plan]


[Example Evaluation for assert odd_uppercase(Cd)==cD]:

"initialize an empty result string" is correct!

"iterate over the string with index" is correct!

"current index is 0" is correct! since we iterate the index from the beginning to the end 

"current index 0 is even", is correct! since 0 is even. "convert the character "C" to lowercase, which is "C" " is incorrect! since the character at index 0 is "C" and the lowercase of "C" is "c" but not "C".  The analysis doesn't correctly convert the character to lowercase when the index is even.  "append "C" to the result string" is correct! but the analysis should append a lowercase "c" to the result string.


[Record]: result string  "C" is correct! since append "C" to the empty string resulting in "C"

"current index is 1" is correct. since the next index of 0 is 1

"current index 1 is odd" is correct since 1 is odd. "convert the character "d" to uppercase, which is "D"" is correct, since the character at index 1 is "d" and the uppercase of "d" is "D".  "append "D" to the result string" is correct!.


[Record]: result string  "CD" is correct! since append "D" to the string "C" resulting in "CD".

"all characters have been processed and appended in their respective cases, return the result string "CD" is correct since all characters have been processed and appended.

In [Results Compare] "The test correct output = cD" is correct! "The logic analysis output = CD" is incorrect! The logic analysis output should be "cD" since the character at index 0 is "C" and the lowercase of "C" is "c" but not "C". The results comparison "cD=CD" is incorrect! The analysis doesn't correctly compare the string with uppercase and lowercase since "cD" is not equal "CD".


The Plan Verification for odd_uppercase(Cd) =cD is incorrect.  The analysis does not correctly convert the character to lowercase when the index is even. The results comparison is incorrect since "cD" is not equal "CD".

[Incorrect Analysis]

[End Example]


Finally, you will be given a problem description starting with [Problem Description], followed by your generated word-described solution plan, starting with [Solution Plan], to solve the [Problem Description]. You will then have one or multiple Logic Verification Processes starting with [Verification for X]  and each applies the solution plan to a test case X. At the beginning of the verification process, [Record analysis] analyses the intermediate variables that should be recorded. During the logic verification process, tag [Record] shows the value updates of the recorded intermediate variable. The  [Results Compare] records the comparison between the logic verification result and the correct test output.

"Let's evaluate the logic analysis" will act as the start to analyse EACH logic verification process, followed by your step-by-step evaluation to verify whether EACH logic verification process is correct or not starting with [Evaluation for X] as shown in examples. Please ONLY evaluate the provided logic verification process. If the logic verification process is correct, the output will be [Correct Analysis], and we will proceed to the next logic verification process. If the  logic verification process is incorrect, an incorrect analysis should be provided and [Inorrect Analysis] will be output to end the analysis.
'''

        self.PY_CODE_GENERATE_SYSTEM = "You are an expert Python Tutor. You teach by Generating the correct interleaved segment of the code plan"

        self.PY_CODE_GENERATE_USER = '''
Finally, You'll receive a Python writing problem starting with [Problem Description]. a verified Global Plan (which already has weights) will be provided，detailing how to solve the problem in a word description.  Then you'll receive a few plan verifications that consider some test cases as input. For each test case X,  the plan verification starting with [Plan Verification for X] considers test case X as input, providing detailed logical reasoning steps and verifying the logical reasoning result against the correct test output, starting with [Results Compare].
Once the plan verification is provided, the "Let's generate the anchor-weighted sequence" flag indicates the start of  writing a sequence of alternating PLAN blocks and CODE blocks.
When generating the program, the plan verification serves as the constraint.  In detail, in the plan verification,  the intermediate variables that should be recorded are analysed at the beginning, starting with [Record analysis] and the value updates of the recorded intermediate variable are clearly shown starting with the  [Record]. It's crucial to ensure the generated program execution remains consistent with the plan verification [Plan Verification for X] when using the same test case X as input.  In other words, when taking the test X as input the generated program should have the same variable value updates as recorded in the plan verification. Additionally, the conditional statements in the generated program should contain all conditions recorded in the plan verifications  [Plan Verification for X] when using the test case X as input. 

**Output Format Requirements:**
1. You must generate a continuous stream of `[GEN_GLOBAL_PLAN]`, `[GEN_PLAN]`, and `[GEN_CODE]` blocks.
2. **Copy the provided Global Plan exactly**, including its `[STEP:N]`
3. **DO NOT** assign weights to the content text (e.g., actual python code or plan descriptions).
6. The output is a sequence of alternating PLAN blocks and CODE blocks.
   • A PLAN block looks like:
     [GEN_PLAN]
     (Current Plan: <one short sentence describing THIS step only>)

   • A CODE block looks like:
     [GEN_CODE]
     <valid Python code for only THIS step>

7. Very important:
   - Only ONE action per plan.
   - Only ONE corresponding code snippet per plan.
   - PLAN describes the intention, CODE implements it.

8. Code blocks must contain **only Python code** and must be syntactically correct.
   - NO reasoning
   - NO extra comments (unless required by syntax)
   - NO references to PLAN

9. DO NOT output any overall summary or explanation—only the stepwise PLAN/CODE alternation.

10.Strict Alignment & Quantity: Ensure every code block aligns perfectly with its specific plan, and the total number of [GEN_CODE] blocks matches exactly with the number of [STEP:x] blocks.
**Output Example:**

[GEN_GLOBAL_PLAN] 
[STEP:1]  STEP:1 description 
[STEP:2]  STEP:2 description 

[GEN_PLAN] 
(Current Plan: plan description.)

[GEN_CODE] 
code description

[GEN_PLAN] 
(Current Plan: plan description.)

[GEN_CODE] 
code description


... (Repeat until finished) ...

[EOS]

"Let's generate the plan code interleaving sequence". Start directly with `[GEN_GLOBAL_PLAN]`.
        '''

        self.PY_CODE_GENERATE_EXAMPLES = '''

[Start Example]
def guess_hat_color(a: str, b: str, c: str, d: str) -> int:\n    \"\"\"\n    # Task\n    Four men, `a, b, c and d` are standing in a line, one behind another.\n    \n    There's a wall between the first three people (a, b and c) and the last one (d).\n    \n    a, b and c are lined up in order of height, so that person a can see the backs of b and c, person b can see the back of c, and c can see just the wall.\n    \n    There are 4 hats, 2 black and 2 white. Each person is given a hat. None of them can see their own hat, but person a can see the hats of b and c, while person b can see the hat of person c. Neither c nor d can see any hats.\n    \n    Once a person figures out their hat's color, they shouts it.\n    \n    ![](http://stuffbox.in/wp-content/uploads/2016/08/Guess-hat-colour-604x270.png)\n    \n    Your task is to return the person who will guess their hat first. You can assume that they will speak only when they reach a correct conclusion.\n    \n    # Input/Output\n    \n    \n    - `[input]` string `a`\n    \n    a's hat color (\"white\" or \"black\").\n    \n    \n    - `[input]` string `b`\n    \n    b's hat color (\"white\" or \"black\").\n    \n    \n    - `[input]` string `c`\n    \n    c's hat color (\"white\" or \"black\").\n    \n    \n    - `[input]` string `d`\n    \n    d's hat color (\"white\" or \"black\").\n    \n    \n    - `[output]` an integer\n    \n    The person to guess his hat's color first, `1 for a, 2 for b, 3 for c and 4 for d`.\n    \"\"\"\n\n

[GEN_GLOBAL_PLAN]
[Algorithm] Rule-based Reasoning.
[STEP:1]  Parse the input strings representing the hat colors of a, b, c, and d.
[STEP:2]  Analyze the visibility constraints: a can see b and c, b can see c, c and d cannot see any hats.
[STEP:3]  Determine the logical deductions each person can make based on the hats they can see and the total number of hats
[STEP:4]  Implement the decision-making process for each person to determine if they can deduce their own hat color.
[STEP:5]  Identify the first person who can correctly deduce their hat color based on the logical deductions.

[GEN_PLAN] 
(Current Plan: Parse the input strings representing the hat colors of a, b, c, and d.)

[GEN_CODE] 
def guess_hat_color(a: str, b: str, c: str, d: str) -> int:

[GEN_PLAN] 
(Current Plan: Determine the logical deductions each person can make based on the hats they can see and the total number of hats.)

[GEN_CODE] 
    if a == b == c == d:
        return 1

[GEN_PLAN] 
(Current Plan: Implement the decision-making process for each person to determine if they can deduce their own hat color.)

[GEN_CODE] 
    if a == b:
        if c != a:          
            return 3
    else:
        return 4

[GEN_PLAN] 
(Current Plan: Identify the first person who can correctly deduce their hat color based on the logical deductions.)

[GEN_CODE] 
    if a != b:
        if b == c:
            return 1
    else: 
        return 2 

[GEN_PLAN] 
(Current Plan: Return the index of the person who guesses their hat color first.)

[GEN_CODE] 
    return 4 

[EOS]
[End Example]
'''

        self.PY_PRINT_GENERATE_SYSTEM = "You are a Python writing assistant that only adds print statements to my current code segment **without removing any comments**."
        self.PY_PRINT_GENERATE_USER = '''
        Finally, you'll receive a Python Program starting with [Python Program]. Then you will be given a few plan verifications for some test cases.  For a test case X, the plan verification, starting with [Plan Verification for X], includes the words description logic to solve the test case X. In the plan verification, the intermediate variables that should be recorded are clearly analysed at the beginning of the verification, starting with [Record analysis] and the updates of intermediate variable values are clearly recorded, starting with [Record].

        "Let's add print statements" flag indicates the start of print statements adding. Then your task is to add the print statements into the provided Python Program to describe how the variables in the program are changed and to ensure the intermediate variable values (described in the plan verification) are printed by the print statement. Please output your program with print statements starting with [Start Program] and ending with [End Program].
        
        **Please ensure that you do not delete any comments in the program, especially those starting with #.**
        '''

        self.PY_PRINT_GENERATE_EXAMPLES = '''
You'll be provided with a few examples structured as follows, beginning with [Start Example] and ending with [End Example]. Within the example, you'll be given a sample Python program, starting with [Example Python Program]. You will be given a few plan verifications for some test cases. For a test case X, its plan verification, starting with [Example Plan Verification for X], includes the words description logic to solve the test case X. In the verification, the intermediate variable that should be recorded is clearly identified starting with [Record analysis] at the beginning of the verification, and the value update of the intermediate variable is clearly recorded, starting with [Record].
Then you will be shown the Python program featuring detailed print statements starting with [Example Python Program with Print Statements]. The print statements are added to describe how the intermediate variable values (described in the plan verification) are changed during the program execution and how the variables in the program are changed. These examples can guide you on where and how to add print statements in the Python program.
**Please ensure that you do not delete any comments in the program, especially those starting with #.**

[Start Example]
# [GEN_GLOBAL_PLAN]
# [STEP:1] Parse the input list of bits representing the binary number.
# [STEP:2] Initialize the output list for the Gray code with the same length as the input list.
# [STEP:3] Set the most significant bit (MSB) of the Gray code to be the same as the MSB of the binary input.
# [STEP:4] Iterate through the remaining bits of the binary input, computing each Gray code bit as the XOR of the current binary bit and the previous binary bit.
# [STEP:5] Return the resulting list of bits representing the Gray code.

# [GEN_PLAN]
# (Current Plan: Parse the input list of bits representing the binary number.)

# [GEN_CODE] 
def bin2gray(bits: list) -> list:

# [GEN_PLAN]
# (Current Plan: Initialize the output list for the Gray code with the same length as the input list.)

# [GEN_CODE]
    gray = [0] * len(bits)

# [GEN_PLAN]
# (Current Plan: Set the most significant bit (MSB) of the Gray code to be the same as the MSB of the binary input.)

# [GEN_CODE]
    gray[0] = bits[0]

# [GEN_PLAN]
# (Current Plan: Iterate through the remaining bits of the binary input, computing each Gray code bit as the XOR of the current binary bit and the previous binary bit.)

# [GEN_CODE]
    for i in range(1, len(bits)):
        gray[i] = bits[i] ^ bits[i - 1]

# [GEN_PLAN]
# (Current Plan: Return the resulting list of bits representing the Gray code.)

# [GEN_CODE] 
    return gray

[Example Plan Verification for assert bin2gray([1, 1]) == [1, 0]]
[Record analysis]
The return value is the list of bits representing the Gray code, so the intermediate Gray code bits should be recorded!

1. Call the function bin2gray([1, 1]).
2. According to line 1 in the solution plan, parse the input list of bits representing the binary number: [1, 1].
3. According to line 2 in the solution plan, initialize the output list for the Gray code with the same length as the input list: [0, 0].
4. According to line 3 in the solution plan, set the most significant bit (MSB) of the Gray code to be the same as the MSB of the binary input: [1, 0].
[Record]: Gray code after setting MSB: [1, 0]
5. According to line 4 in the solution plan, iterate through the remaining bits of the binary input:
   - For the second bit (index 1): compute Gray code bit as XOR of current binary bit (1) and previous binary bit (1): 1 XOR 1 = 0. Update Gray code: [1, 0].
   [Record]: Gray code after processing bit at index 1: [1, 0]
6. According to line 5 in the solution plan, return the resulting list of bits representing the Gray code: [1, 0].

[Example Python Program with Print Statements]
# [GEN_GLOBAL_PLAN]
# [STEP:1] Parse the input list of bits representing the binary number.
# [STEP:2] Initialize the output list for the Gray code with the same length as the input list.
# [STEP:3] Set the most significant bit (MSB) of the Gray code to be the same as the MSB of the binary input.
# [STEP:4] Iterate through the remaining bits of the binary input, computing each Gray code bit as the XOR of the current binary bit and the previous binary bit.
# [STEP:5] Return the resulting list of bits representing the Gray code.

# [GEN_PLAN]
# (Current Plan: Parse the input list of bits representing the binary number.)

# [GEN_CODE] 
def bin2gray(bits: list) -> list:
    print(f"Input list of bits representing the binary number: {bits}")

# [GEN_PLAN]
# (Current Plan: Initialize the output list for the Gray code with the same length as the input list.)

# [GEN_CODE]
    gray = [0] * len(bits)
    print(f"Initialized Gray code list with the same length as input: {gray}")

# [GEN_PLAN]
# (Current Plan: Set the most significant bit (MSB) of the Gray code to be the same as the MSB of the binary input.)

# [GEN_CODE]
    gray[0] = bits[0]
    print(f"Gray code after setting MSB: {gray}")

# [GEN_PLAN]
# (Current Plan: Iterate through the remaining bits of the binary input, computing each Gray code bit as the XOR of the current binary bit and the previous binary bit.)

# [GEN_CODE]
    for i in range(1, len(bits)):
        gray[i] = bits[i] ^ bits[i - 1]
        print(f"Gray code after processing bit at index {i}: {gray}")

# [GEN_PLAN]
# (Current Plan: Return the resulting list of bits representing the Gray code.)

# [GEN_CODE] 
    print(f"Returning the resulting list of bits representing the Gray code: {gray}")
    return gray

[End Example]
'''

        self.PY_PROGRAM_EXPLAIN_SYSTEM = self.PY_PROGRAM_EXPLAIN_SYSTEM = (
    "You are a meticulous Python Code Interpreter and Explainer. "
    "Your task is to read the provided Python program, which includes reasoning plans ([GEN_PLAN]) and code segments ([GEN_CODE]). "
    "Please explain the execution flow and the semantic meaning of each line of code in plain English. "
    "Do not assess whether the code is correct or incorrect yet; simply describe objectively what the code is doing line by line."
)

        self.PY_PROGRAM_EXPLAIN_USER = '''
You will be provided with an example of a structured coding solution, starting with [Example Structured Solution]. This solution contains Global Plans, Steps, Local Plans, and Code.
You'll be provided with a few examples, each starting with [Start Example] and ending with [End Example]. In each example, you will be given an example Python programming problem starting with [Example Problem Description] and also an example Python program, marked as [Example Python Program] generated for the Python programming problem.  

1. `[GEN_GLOBAL_PLAN]`: The reasonable Global Solution Plan to each step，which is used to guide the generation of subsequent local plans and local code
2. `[GEN_PLAN]`: A local plan used to guide the reasoning or intention behind subsequent local code
3. `[GEN_CODE]`: The actual Python code implementation.

Your goal is to generate a **Line-by-Line Explanation** for the code. 
- For `[GEN_GLOBAL_PLAN]` lines: Summarize the intent.
- For `[GEN_PLAN]` lines: Summarize the intent.
- For `[GEN_CODE]` lines: Explain exactly what the Python interpreter does (e.g., "Assigns variable X", "Checks condition Y").

Below is a generic example of the input and the expected output format.You don't need to analyze empty lines

[Start Example]

[Example Problem Description]
def reformat(s: str) -> str:\n \"\"\"\n Given alphanumeric string s. (Alphanumeric string is a string consisting of lowercase English letters and digits).\n You have to find a permutation of the string where no letter is followed by another letter and no digit is followed by another digit. That is, no two adjacent characters have the same type.\n Return the reformatted string or return an empty string if it is impossible to reformat the string.\n \n Example 1:\n Input: s = \"a0b1c2\"\n Output: \"0a1b2c\"\n Explanation: No two adjacent characters have the same type in \"0a1b2c\". \"a0b1c2\", \"0a1b2c\", \"0c2a1b\" are also valid permutations.\n \n Example 2:\n Input: s = \"leetcode\"\n Output: \"\"\n Explanation: \"leetcode\" has only characters so we cannot separate them by digits.\n \n Example 3:\n Input: s = \"1229857369\"\n Output: \"\"\n Explanation: \"1229857369\" has only digits so we cannot separate them by characters.\n \n Example 4:\n Input: s = \"covid2019\"\n Output: \"c2o0v1i9d\"\n \n Example 5:\n Input: s = \"ab123\"\n Output: \"1a2b3\"\n \n \n Constraints:\n \n 1 <= s.length <= 500\n s consists of only lowercase English letters and/or digits.\n \"\"\"\n

[Example Python Program]
[GEN_GLOBAL_PLAN] 
[STEP:1] Parse the input string `s` to separate the characters into two lists: one for digits and one for letters.
[STEP:2] Check the lengths of the two lists. If the absolute difference between the lengths of the two lists is greater than 1, return an empty string as it is impossible to reformat.
[STEP:3] Determine which list (digits or letters) should start the reformatted string based on their lengths.
[STEP:4] Interleave the characters from the two lists to form the reformatted string, ensuring no two adjacent characters are of the same type.
[STEP:5] Return the reformatted string as the result.

[GEN_PLAN]
(Current Plan: Parse the input string `s` to separate the characters into two lists: one for digits and one for letters.)

[GEN_CODE] 
def reformat(s: str) -> str:

[GEN_PLAN]
(Current Plan: Check the lengths of the two lists. If the absolute difference between the lengths of the two lists is greater than 1, return an empty string as it is impossible to reformat.)

[GEN_CODE] 
    if abs(len(digits) - len(letters)) > 1:
        return \"\"
        
[GEN_PLAN]
(Current Plan: Determine which list (digits or letters) should start the reformatted string based on their lengths.)

[GEN_CODE] 
    if len(digits) > len(letters):
        longer, shorter = digits, letters
            else:
                longer, shorter = letters, digits

[GEN_PLAN]
(Current Plan: Interleave the characters from the two lists to form the reformatted string, ensuring no two adjacent characters are of the same type.)

[GEN_CODE] 
    result = []
        for i in range(len(s)):
            if i % 2 == 0:
                result.append(longer[i // 2])
            else:
                result.append(shorter[i // 2])

[GEN_PLAN]
(Current Plan: Return the reformatted string as the result.)

[GEN_CODE] 
    return \"\".join(result)

[Example Explanation For Each Line]
Line 1:[GEN_GLOBAL_PLAN] 
Explanation:Indicates the start of the overall high-level solution strategy. The importance weight of this global plan is 7.

Line 2:[STEP:1] Parse the input string...
Explanation:Defines Step 1 of the global plan: separating digits and letters into two different lists.

Line 3:[STEP:2] Check the lengths...
Explanation:Defines Step 2: validate whether interleaving is possible by checking if the difference in list lengths is ≤ 1.

Line 4:[STEP:3] Determine which list should start...
Explanation:Defines Step 3: decide which type (digit or letter) should appear first based on which list is longer.

Line 5:[STEP:4]  Interleave the characters...
Explanation:Defines Step 4: the merging / alternating process between digits and letters.

Line 6:[STEP:5] Return the reformatted string...
Explanation:Defines Step 5: return the final combined string.

Line 7:[GEN_PLAN]
Explanation:Marks the beginning of a local plan with high importance.

Line 8:(Current Plan: Parse the input string...)
Explanation:Describes the goal for the next code segment: split s into two lists (digits and letters).

Line 9:[GEN_CODE] 
Explanation:Indicates that code follows to implement the local plan.

Line 10:def reformat(s: str) -> str:
Explanation:Defines a function named reformat that takes a string s and returns a string. This line establishes the function scope.

Line 11:[GEN_PLAN]
Explanation:Starts a new local plan with high importance.

Line 12:(Current Plan: Check the lengths...)
Explanation:States the goal of the next code block: validate whether reforming is possible by comparing lengths of the two lists.

Line 13:[GEN_CODE] 
Explanation:Marks the code implementing the checking logic.

Line 14:if abs(len(digits) - len(letters)) > 1:
Explanation:In function scope, evaluates the absolute difference between the sizes of digits and letters.If the difference is greater than 1, interleaving becomes impossible.

Line 15:return ""
Explanation:If the condition is true, immediately returns an empty string, ending the function execution.

Line 16:[GEN_PLAN]
Explanation:Begins another local plan.

Line 17:(Current Plan: Determine which list should start...)
Explanation:States that next code will decide which type (digit or letter) appears first depending on which list is longer.

Line 18:[GEN_CODE] 
Explanation:Indicates incoming code related to selecting the starting list.

Line 19:if len(digits) > len(letters):
Explanation:Checks whether the digit list has more elements than the letter list.

Line 20:longer, shorter = digits, letters
Explanation:If digits are more numerous, assigns digits to the variable longer and letters to the variable shorter.

Line 21:else:
Explanation:Begins the alternative branch when letters have equal or greater count.(NOTE: indentation is described objectively; no correctness judgment.)

Line 22:longer, shorter = letters, digits
Explanation:Assigns letters as the longer list and digits as the shorter list in the else-branch.

Line 23:[GEN_PLAN]
Explanation:Starts a new local plan.

Line 24:(Current Plan: Interleave the characters...)
Explanation:States intention: merge both lists in alternating positions.

Line 25:[GEN_CODE] 
Explanation:Marks the beginning of the interleaving code block.

Line 26:result = []
Explanation:Initializes an empty list result which will store the alternated characters in order.

Line 27:for i in range(len(s)):
Explanation:Starts a loop running from i = 0 to i = len(s) - 1, iterating exactly once per expected output character.

Line 28:if i % 2 == 0:
Explanation:Checks whether the index i is even.

Line 29:result.append(longer[i // 2])
Explanation:If the index is even, appends an element from the longer list at position i // 2 to result.

Line 30:else:
Explanation:Marks the branch for odd indexes.

Line 31:result.append(shorter[i // 2])
Explanation:If i is odd, appends a character from the shorter list at index i // 2.

Line 32:[GEN_PLAN]
Explanation:Marks a new local plan describing the final step.

Line 33:(Current Plan: Return the reformatted string...)
Explanation:The goal of the next code piece is simply to return the final joined string.

Line 34:[GEN_CODE] 
Explanation:Indicates the final code block.

Line 35:return "".join(result)
Explanation:Combines all strings inside result into one continuous string and returns it as the function's final output.

[End Example]

Finally, you'll be presented with a problem description, starting with [Problem Description] and A text containing Global plan, steps, and interleaved local plans and local codes, starting with [Python Program] to solve the [Problem Description].  Following this, the "Let's generate the explanation" flag will signal the start of the explanation phase. Your task is to generate the word explanation for each line in the Python Program following the examples shown before. Please skip the explanation for the program line which is a print statement. 
Please output your explanation starting with [Start Explanation] and ending with [End Explanation].
'''

        self.PY_PROGRAM_ANALYSIS_SYSTEM = """
        You are a Hierarchical Code Auditor.You will be given two logical reasoning processes [Correct Logic Reasoning Process] and [Error Execution Trace]. Your task is to identify any errors in [Error Execution Trace] by comparing it with the [Correct Logic Reasoning Process].

Your task is to analyze the failure on three levels:
1. **Plan Validity**: Is the Local Plan correct and consistent with the Global Plan?
2. **Implementation Alignment**: Did the Code actually implement what the Local Plan claimed?
3. **Execution Logic**: Does the actual execution trace deviate from the Correct Logic Reasoning Process?
You must scan the program from top to bottom. For each block (Global Plan -> Local Plan -> Code), you perform a check. If you find an error (inconsistency, wrong logic, or bad alignment), you must immediately raise a `[risk]` flag, explain the problem, and provide the solution (how to fix the plan or the code).
Finally, you will summarize your findings in a structured analysis.
        """
        self.PY_PROGRAM_ANALYSIS_USER = '''
You will be provided with the following inputs:
1. **[Problem Description]**: The task requirements.
2. **[Error Program with Plans]**: The interleaved text containing [GEN_GLOBAL_PLAN], [GEN_PLAN] (Local Plan), and [GEN_CODE] (Local Code) to solve the [Problem Description].
3. **[Error Execution Trace]**: A detailed execution trace, including intermediate variable values, for the failed test case X, starting with [Error Execution Trace for Test Case X].
4. **[Correct Program]**: The ground truth / golden solution (Interleaved Code).**Please note, if the correct program contains class like class solution, please disregard it's class definition and refer to the function logic within it and maintain the original function format**.

### Your Task:

**Phase 1: Sequential Audit (Output under [Compare Results])**
Read the [Error Program with Plans] sequentially. For each part, compare it with the [Problem Description] and logic derived from [Correct Program].

- **Check Global Plan ([GEN_Global_PLAN])**:
    -the local plan starts with [GEN_Global_PLAN],you should analyze whether this plan is consistent with the strategy used in the [Correct Program]
- **Check Local Plan ([GEN_PLAN])**: 
   - the local plan starts with [GEN_PLAN],you should analyze whether this plan is consistent with the strategy used in the [Correct Program] and the Global Plan? Is the logic correct?
   - Example: If the Correct Program uses addition, but the Local Plan says "multiply", the plan is wrong.
- **Check Code ([GEN_CODE])**: 
   - Analyze whether it follows the Local Plan.
   - Analyze whether the execution logic matches the Correct Program.
   
**If you find an error at any step, output a risk block immediately:**
If there is a problem with the plan， you need analyze how to modify the plan.
If there is a problem with the code， you need analyze how to modify the code.
Format:
<Location>
[risk]：Explain why this is wrong (e.g., contradicts global plan, wrong operator, logic error).
[solution]:How to modify the code or plan.

**Phase 2: Final Summary (Output under [My Analysis])**
- **Summary**: List the risks and solution found in Phase 1 concisely.
- **Execution Logic Error Analysis**: Provide a holistic analysis of why the execution failed, tracing the root cause from the first error to the final wrong output.

**Constraint**: Do NOT output the full corrected program. Only output the analysis and fixes in text.

For every error found, output a block starting with location (e.g., "Local Plan 2" or "Code Segment 3") , followed by the`[risk]` and '[solution]'on separate lines.
starting with [My Analysis]

Example of Locations:"Local Plan 1", "Code Segment 1", "Local Plan 2", "Code Segment 2", etc.
'''

        self.PY_PROGRAM_ANALYSIS_EXAMPLES = '''
You'll be provided with a few examples, each starting with [Start Example] and ending with [End Example]. In each example, you will be given an example Python programming problem starting with [Problem Description] and also an example of an error Python program, marked as [Error Program] generated for the Python programming problem. For a failed test case X, you'll receive a detailed execution trace of the example error program marked as [Error Execution Trace for Test Case X], including intermediate variable values. 
Additionally, you'll be provided with and **a correct program (code only)**, marked as [Correct Program]. Please study this correct program carefully and reflect on the existing incorrect program based on this correct program.**Please note, if the correct program contains class like class solution, please disregard it's class definition and refer to the function logic within it and maintain the original function format**. Subsequently, [Compare Results] describes the process of comparing the Example Correct Logic Reasoning Process with the Example Error Execution Trace, elucidating the differences in their outputs and pinpointing where the Error Execution Trace deviates from correctness.
Lastly, [My Analysis] concludes the errors in the [Compare Results] and proposes solutions to rectify these errors.

[Start Example]
[Problem Description]
Calculate the average of a list of numbers. Return 0 if list is empty.

[Error Program with plan]
[GEN_GLOBAL_PLAN] 
[STEP:1] Handle empty list case. 
[STEP:2] Sum all numbers. 
[STEP:3] Divide sum by length.
[STEP:4] Return the sum.

[GEN_PLAN]
(Current Plan: Check if the list is empty. If it is, return -1 to indicate error.)

[GEN_CODE] 
def calculate_average(nums):
    if len(nums) == 0:
        return 0

[GEN_PLAN]
(Current Plan: Loop through nums and multiply elements.)

[GEN_CODE] 
    total = 0
    for x in nums:
        total = total * x

[GEN_PLAN]
(Current Plan: return total.)

[GEN_CODE] 
    return total

[Error Execution Trace for Test Case nums=[2, 3]]
1. len(nums) is 2 != 0.
2. Initialize total = 0.
3. x=2, total = 0 * 2 = 0.
4. x=3, total = 0 * 3 = 0.
5. Return 0.

[Correct Program]
def calculate_average(nums):
    if not nums:
        return 0
    return sum(nums) / len(nums)

Let's do analysis
[Compare Results]

Starting audit of the Error Program...

1. **Analyzing [GEN_PLAN] (Local Plan 1)**: "(Current Plan: Check if list is empty. If so, return -1.)"
   - Comparison: The Problem Description explicitly says "Return 0". The Correct Program also returns 0 (`return 0`). This plan says "return -1".
   - Status: Error found.
   
   Local Plan 1
   [risk]: The plan contradicts the requirement to return 0 for empty lists (and the Correct Program logic).
   [solution]: Change plan to "Check if list is empty. If so, return 0."

2. **Analyzing [GEN_CODE] (Code Segment 1)**: `if len(nums) == 0: return 0`
   - Comparison: The code returns 0, which aligns with the Correct Program, but it ignored the (wrong) Local Plan 1.
   - Status: Correct logic, but inconsistent with its local plan.

3. **Analyzing [GEN_PLAN] (Local Plan 2)**: "(Current Plan: Loop through nums and multiply elements.)"
   - Comparison: The Correct Program uses `sum(nums)` which implies addition. This plan says "multiply".
   - Status: Error found.
   
   Local Plan 2
   [risk]: The plan suggests multiplication, but the Correct Program logic requires summation (addition).
   [solution]: Change plan to "Loop through nums and add elements to total."

4. **Analyzing [GEN_CODE] (Code Segment 2)**: `total = 0; total = total * x`
   - Comparison with Correct Logic: Correct Program implies addition (`2+3=5`). Execution trace shows `0*2=0`.
   - Status: Error found.
   
   Code Segment 2
   [risk]: The code implements multiplication (following the wrong plan) instead of addition. Also, initializing `total=0` for multiplication results in permanent zero.
   [solution]: Change `total = total * x` to `total = total + x`.

5. **Analyzing [GEN_PLAN] (Local Plan 3)**: "(Current Plan: Return total.)"
   - Comparison: Correct Program uses `sum(nums) / len(nums)`. This plan just returns total.
   - Status: Error found.
   
   Local Plan 3
   [risk]: The plan misses the division step required to calculate the average.
   [solution]: Change plan to "Divide total by the length of nums and return the result."

6. **Analyzing [GEN_CODE] (Code Segment 3)**: `return total`
   - Comparison: Code returns the sum (actually 0 due to previous bug) instead of average.
   - Status: Error found.
   
   Code Segment 3
   [risk]: The code returns the accumulated value directly without dividing by length.
   [solution]: Change `return total` to `return total / len(nums)`.

[My Analysis]
<Local Plan 1>
[risk]: The plan contradicts the requirement to return 0 for empty lists.
[solution]: Change plan to "Check if list is empty. If so, return 0."
<Local Plan 2>
[risk]: The plan suggests multiplication, but the Correct Program logic requires summation (addition).
[solution]: Change plan to "Loop through nums and add elements to total."
<Code Segment 2>
[risk]: The code implements multiplication (following the wrong plan) instead of addition. Also, initializing `total=0` for multiplication results in permanent zero.
[solution]: Change `total = total * x` to `total = total + x`.
<Local Plan 3>
[risk]: The plan misses the division step required to calculate the average.
[solution]: Change plan to "Divide total by the length of nums and return the result."
<Code Segment 3>
[risk]: The code returns the accumulated value directly without dividing by length.
[solution]: Change `return total` to `return total / len(nums)`.

Execution Logic Error Analysis:
The execution failed primarily because the logic deviated from the Correct Program starting at Local Plan 2. The code attempted to calculate a product instead of a sum (`total * x`), which contradicts the `sum()` logic in the Correct Program. Combined with an initial value of 0, this caused the `total` variable to remain 0. Furthermore, the final step failed to perform the division operation required for an average calculation (`/ len(nums)`).

[End Example]
'''

        self.PY_PROGRAM_CORRECT_SYSTEM = '''
You are a Hierarchical Program Repairer. Your task is to fix a structured program (consisting of Global Plans, Local Plans, and Code) based on a provided Error Analysis Report.

You must:
1. Read the [Error Analysis] carefully, focusing on lines marked with `[risk]`.
2. Fix the **Local Plans** if the analysis says they are invalid or inconsistent.
3. Fix the **Code** if the analysis says it implements wrong logic or aligns poorly with the plan.
4. Ensure the final output maintains the strict interleaved format: `[GEN_GLOBAL_PLAN]` -> `[GEN_PLAN]` -> `[GEN_CODE]`.
5. Keep existing valid logic and comments (especially #) unchanged.

        '''
        self.PY_PROGRAM_CORRECT_USER = '''
You will be provided with:
1. **[Problem Description]**: The task requirements.
2. **[Error Program]**: The original wrong program with plans.
3. **[Error Analysis]**: A report containing `[risk]` tags identifying specific locations (Plans or Code) and the required fixes.

### Your Task:

**Step 1: Apply Fixes**
- Locate the specific Plan or Code segment mentioned in each `[risk]` tag.
- Apply the suggested "Plan Fix" to the `[GEN_PLAN]`.
- Apply the suggested "Code Fix" to the `[GEN_CODE]`.
- If a part is NOT mentioned in the [Error Analysis], keep it exactly as it is (do not rewrite correct parts).

**Step 2: Generate Output**
- Output the full **[Fixed Program]** in the correct format.
- Output a summary **[Explanation Adjustments]** listing exactly what you changed and why.

**Format Constraints:**
- The [Fixed Program] must be a valid, runnable Python script mixed with Plan comments.
- Please ensure that you have completed the modifications to the error program while providing Explanation Adjustments, rather than just giving Explanation Adjustments without modifying the error program

Finally, you'll receive a few examples structured as follows, starting with [Start Example] and ending with [End Example]. Within each example, you'll find a Python programming problem beginning with [Example Problem Description], followed by an error program presented under [Example Error Program] for the program problem. 
Additionally, you'll be given the [Example Fixing Analysis] locating which lines in the error program lead to errors by step-by-step analysis of the [Example Error Program] and some suggestions to fix the program. 
Then you will be given the corrected Python program under [Example Fixed Program], aligned with the error analysis and fixing analysis. Then the explanation adjustment, starting with [Example Explanation Adjustments] is given to display which program lines are changed and explain the reason why it is changed.

[Start Example]
[Problem Description]
Write a function `solve` that returns the sum of a list. Return 0 if empty.

[Example Error Program]
[GEN_GLOBAL_PLAN] 
[STEP:1] Handle empty inputs. 
[STEP:2] Calculate sum. 
[STEP:3] Return result.

[GEN_PLAN]
(Current Plan: Check if list is empty. If so, return -1.)

[GEN_CODE] 
def solve(nums):
    if len(nums) == 0:
        return 0

[GEN_PLAN] 
(Current Plan: Multiply all elements.)

[GEN_CODE] 
    total = 0
    for x in nums:
        total = total * x

[GEN_PLAN]
(Current Plan: Return total.)

[GEN_CODE] 
    return total

[Example Error Analysis]
<Local Plan 1>
[risk] : The plan contradicts the requirement "Return 0". 
[solution]: Change "return -1" to "return 0".

<Code Segment 1>
[risk] : Code followed the global plan (returning 0) but ignored the (wrong) local plan. 
[solution]: Code is actually correct logic-wise, but let's keep it aligning with the fixed plan.

<Local Plan 2>
[risk] : The plan suggests multiplication, but the task requires sum. 
[solution]: Change to "Add all elements".

<Code Segment 2>
[risk] : Code implements multiplication (`*`) and initializes `total=0`. 
[solution]: Change operator to `+` and keep `total=0` (valid for sum).

Execution Logic Error Analysis:
The execution failed because the implementation began to diverge from the intended global objective starting at Local Plan 2. Although Code Segment 1 correctly returned 0 for empty lists, the associated Local Plan 1 incorrectly stated the intention to return -1, creating a logical inconsistency between plan and code. While this did not break execution, it reflects a mismatch between planning and implementation.The major error occurred in Local Plan 2, where the plan incorrectly specified multiplication instead of addition. This misdirected the developer intent and directly caused Code Segment 2 to use the operator * instead of +. Because total was initialized to 0, multiplying any value with zero resulted in total remaining zero for the entire loop, making the computation incorrect. The operation should have been addition, which aligns with the problem requirement of computing a sum.Overall, the primary breakdown came from the incorrect specification in Local Plan 2, leading the corresponding code to perform multiplication instead of addition. Correcting both the plan and the operator restores alignment with the task objective.

[Example Fixing Analysis]
1. <Fixing Local Plan 1>: The original plan said "return -1". I will change it to "return 0" to match the Problem Description.
2. <Fixing Local Plan 2>: The original plan said "Multiply". I will change it to "Add all elements".
3. <Fixing Code Segment 2>: The original code used `*`. I will change it to `+` to match the sum logic.

[Example Fixed Program]
[GEN_GLOBAL_PLAN] 
[STEP:1] Handle empty inputs. 
[STEP:2] Calculate sum. 
[STEP:3] Return result.

[GEN_PLAN]
(Current Plan: Check if list is empty. If so, return 0.)

[GEN_CODE]
def solve(nums):
    if len(nums) == 0:
        return 0

[GEN_PLAN]
(Current Plan: Iterate through the list and add all elements to total.)

[GEN_CODE]
    total = 0
    for x in nums:
        total = total + x

[GEN_PLAN]
(Current Plan: Return total.)

[GEN_CODE] 
    return total

[Example Explanation Adjustments]
1. <Local Plan 1>: Modified plan text from "return -1" to "return 0" to align with the global plan.
2. <Local Plan 2>: Modified plan text from "Multiply all elements" to "Iterate through the list and add all elements" to correct the logic error.
3. <Code Segment 2>: Changed `total = total * x` to `total = total + x` to implement the summation logic correctly.

[End Example]

You'll encounter a Python writing problem starting with [Problem Description]. You will be given a program fixing history starting with  [Incorrect History]. This history includes all error programs generated before and each error program starting with [History Error Program]. When you generate a new program, please avoid generating the same error programs.  Following that, the latest error program will be presented under [Error Program]. Then you will be given the explanation for the error program including an explanation for each program line starting with [Error Program Explanation]. Subsequently, you'll receive an error analysis, starting with [Error Analysis], describing the error in the error program. The repair process will begin with "Let's correct the program". Then please follow the fixing analysis, shown in the example to generate the fixing analysis, starting with [Fixing Analysis] to detail analysis of which lines in the error program lead to errors by step-by-step analysis of the [Error Program Explanation] and then generate some suggestions to fix the program. Please give the suggestions as specific as possible.   Then please generate your repaired program based on the fixing analysis and error analysis. Please generate your fixed program starting with [Start Fixed Program] and ending with [End Fixed Program] and  Please ONLY include Python Program between [Start Fixed Program] and [End Fixed Program]. Finally, provide your explanation adjustments, starting with [Explanation Adjustments], to elucidate how the program is altered to align with the fixing analysis and error analysis. 
'''