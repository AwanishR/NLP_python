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
#print ("File start position:",noteFile.tell())
tokenCount =0
for line in noteFile:
    print("Original note Text read from file\n",line)
    #Sentence Detection
    sentences = sentence_detector.tokenize(line.strip())
    #print ("\nDetected Sentences from above note")
    for index in range (len(sentences)):
        fileToWrite.write("\nSentence"+str(index+1)+" >>> "+sentences[index]+" <<<\n\nTokens of this sentence are as follows\n")
        #print ("\n++++++\n"+sentences[index])
        #Tokenization
        tokens = word_tokenize(sentences[index])
        #POS Tagging [PennTreebank tagger]
        postag=pos_tag(tokens);
        #print ("Post Tagg")
        #print(postag)
        for item in tokens:
            fileToWrite.write("\n----\n{}".format(item))
            tokenCount+=len(item)
        fileToWrite.write("\n*****POS Tagging [using Penn Treebank tagging]****\n")
        for item in postag:
            fileToWrite.write("{} ".format(item))
        fileToWrite.write("\n")
        #print("\nDetected Tokens\n")
        #print list items iteratively
        #print ("\n-----\n".join(tokens))
fileToWrite.close()
print ("\nTotal number of sentences detected :",len(sentences))
print("\nTotal number of tokens detected :",tokenCount)
#print ("\nFile end position:",noteFile.tell())
noteFile.close()
print ("\nProcessing completed!!!")

                
