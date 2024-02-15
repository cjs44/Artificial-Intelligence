"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Decrease noise from 0.2 to 0.01
    This makes it more likely it will take the right action
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise

def question3a():
    """
    Prefer the close exit (+1), risking the cliff (-10)
    Decrease noise so it is fine risking the cliff
    Decrese the discount and living reward so it chooses the closer exit
    """

    answerDiscount = 0.6
    answerNoise = 0.1
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10)
    Increase the noise so it avoids the cliff
    Decrease the discount and living reward so it chooses the closer exit
    """

    answerDiscount = 0.6
    answerNoise = 0.3
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Prefer the distant exit (+10), risking the cliff (-10)
    Decrease noise so it risks the cliff
    Increase the living reward so it chooses the far exit
    """

    answerDiscount = 0.8
    answerNoise = 0.01
    answerLivingReward = 1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10)
    Increase living reward and decrease discount to choose far exit
    Increase noise to avoid cliff
    """

    answerDiscount = 0.5
    answerNoise = 0.3
    answerLivingReward = 1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Avoid both exits (also avoiding the cliff)
    Increase noise to avoid cliff
    Increase living reward so it tries to stay alive
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 2.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Tried many combinations of the two and cannot find a solution
    """

    # answerEpsilon = 0.3
    # answerLearningRate = 0.5

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
