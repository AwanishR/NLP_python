import nltk
from nltk import word_tokenize
from nltk import pos_tag

#Open sentence detector model
sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
#Open notes file for reading
noteFile = open ("f:/NLP/data/sample_note_1.txt","r")
#Opens output file for writing
fileToWrite = open("processednotes.txt","w")
print ("Processing note of file ",noteFile.name,"....")
tokenCount =0
sentCount=0
for line in noteFile:
    print("Original note Text read from file\n",line)
    #Sentence Detection
    sentences = sentence_detector.tokenize(line.strip())
    for index in range (len(sentences)):
        fileToWrite.write("\nSentence"+str(index+1)+" >>> "+sentences[index]+" <<<\n\nTokens of this sentence are as follows\n")
        #Tokenization
        tokens = word_tokenize(sentences[index])
        #POS Tagging [PennTreebank tagger]
        postag=pos_tag(tokens);
        #Write tokens into file
        for item in tokens:
            fileToWrite.write("\n----\n{}".format(item))
            tokenCount+=len(item)
        fileToWrite.write("\n*****POS Tagging [using Penn Treebank tagging]****\n")
        #Write POS tag in file
        for item in postag:
            fileToWrite.write("{} ".format(item))
        fileToWrite.write("\n")
    sentCount+=len(sentences)
fileToWrite.close()
#Dsiplays
print ("\nTotal number of sentences detected :",sentCount)
print("\nTotal number of tokens detected :",tokenCount)
noteFile.close()
print ("\nProcessing completed!!!")

                
