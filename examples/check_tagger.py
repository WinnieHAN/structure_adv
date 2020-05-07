# from nltk.tag import CRFTagger
# ct = CRFTagger()
# train_data = [[('University','Noun'), ('is','Verb'), ('a','Det'), ('good','Adj'), ('place','Noun')],[('dog','Noun'),('eat','Verb'),('meat','Noun')]]
# ct.train(train_data,'model.crf.tagger')
# ct.tag_sents([['dog','is','good'], ['Cat','eat','meat']])
# # [[('dog', 'Noun'), ('is', 'Verb'), ('good', 'Adj')], [('Cat', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]
# gold_sentences = [[('dog','Noun'),('is','Verb'),('good','Adj')] , [('Cat','Noun'),('eat','Verb'), ('meat','Noun')]]
# ct.evaluate(gold_sentences)



# from nltk.tag import HunposTagger
# ht = HunposTagger('en_wsj.model')
# ht.tag('What is the airspeed of an unladen swallow ?'.split())
# # [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'NN'), ('swallow', 'VB'), ('?', '.')]
# ht.close()

from nltk.tag.senna import SennaTagger
st = SennaTagger('/home/hanwj/PycharmProjects/structure_adv/tagging_models/senna')
text = "The quick brown fox jumps over the lazy dog"
a = st.tag(text.split())
print(a)
# [('The', u'DT'), ('quick', u'JJ'), ('brown', u'JJ'), ('fox', u'NN'), ('jumps', u'VBZ'), ('over', u'IN'), ('the', u'DT'), ('lazy', u'JJ'), ('dog', u'NN')]

from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger(model_filename='/home/hanwj/PycharmProjects/structure_adv/tagging_models/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger', path_to_jar='/home/hanwj/PycharmProjects/structure_adv/tagging_models/stanford-postagger-2018-10-16/stanford-postagger.jar')
a = st.tag('What is the airspeed of an unladen swallow ?'.split())
# [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
print(a)