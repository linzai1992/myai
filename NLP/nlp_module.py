import spacy

class NLPModule:
	def __init__(self):
		self.nlp = spacy.load("en")

	def process_command(self, command):
		doc = self.nlp(command)
		print("Parse..........")
		for word in doc:
			print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

		print("Entities.......")
		for ent in doc.ents:
			print(ent.label_, ent.text)
		

m = NLPModule()
while True:
	i = input("Enter command: ")
	m.process_command(i)