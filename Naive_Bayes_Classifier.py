#creating a spam model
#train
def train():
    total = 0
    numSpan = 0
    for email in trainData:
        if(email.label == SPAM):
            numSpan += 1
        total += 1
        processEmail(email.body, email.label)
        pA = numSpan/float(total)
        pNotA = (total - numSpan)/float(total)

#reading words from a specific label
def processEmail(body, label):
    for word in body:
        if(label == SPAM):
            trainPositive[word] = trainPositive.get(word, 0) + 1
            positiveTotal += 1
        else:
            trainNegative[word] = trainNegative.get(word, 0) + 1
            negativeTotal += 1

#gives conditional probability
def conditionalEmail(body, spam):
    result = 1.0
    for word in body:
        result *= conditionalWord(body, spam)
    return result

#classifies a new email as spam/not spam
def classify(email):
    isSpam = pA * conditionalEmail(email, True) #P(A | B)
    notSpam = pNotA * conditionalEmail(email, False) # P(¬A | B)
    return isSpam > notSpam

#laplace soothing for words not present in dict
def conditionalWord(word, spam):
    if spam:
        return((trainPositive.get(word, 0) + alpha)/((float)(positiveTotal + alpha*numWords)))
    return ((trainNegative.get(word, 0) + alpha) / ((float)(negativeTotal + alpha * numWords)))

